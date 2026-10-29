"""FHI-aims jobs for molecular infrared spectra."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from jobflow import Flow, Response, job
from monty.json import MSONable
from monty.serialization import dumpfn
from pymatgen.io.aims.sets.core import StaticSetGenerator

from atomate2.aims.jobs.base import BaseAimsMaker
from atomate2.aims.run import run_aims, should_stop_children
from atomate2.aims.schemas.molecular_ir import (
    ImaginaryModeCorrection,
    MolecularIRTaskDoc,
)
from atomate2.aims.schemas.task import AimsTaskDoc
from atomate2.common.files import gzip_output_folder

if TYPE_CHECKING:
    from pymatgen.core import Molecule
    from pymatgen.io.aims.sets.base import AimsInputGenerator

    from atomate2.aims.schemas.molecular_ir import SpectrumSettings


@dataclass(frozen=True)
class MolecularIRSettings(MSONable):
    """IR analysis settings and shared FHI-aims input parameters."""

    displacement: float = 0.0025
    imaginary_threshold_cm_1: float = -20.0
    imaginary_displacement_angstrom: float = 0.10
    max_imaginary_cycles: int = 3
    user_params: dict[str, Any] = field(
        default_factory=lambda: {"xc": "b3lyp", "species_dir": "tight"}
    )


def displace_imaginary_mode(
    document: MolecularIRTaskDoc,
    maximum_displacement: float,
    mode_index: int,
) -> Molecule:
    """Displace a copy along the selected normal mode for correction."""
    if not np.isfinite(maximum_displacement) or maximum_displacement <= 0:
        raise ValueError("Maximum imaginary-mode displacement must be positive")
    if not 0 <= mode_index < len(document.normal_modes):
        raise IndexError("Invalid imaginary-mode index")
    mode = np.asarray(document.normal_modes[mode_index], dtype=float)
    norm = np.max(np.linalg.norm(mode, axis=1))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Imaginary mode has no usable displacement vector")
    displaced = deepcopy(document.molecule)
    for atom, vector in enumerate(maximum_displacement * mode / norm):
        displaced.translate_sites([atom], vector)
    return displaced


@dataclass
class MolecularAimsMaker(BaseAimsMaker):
    """Run FHI-aims on a Molecule and return a standard AimsTaskDoc."""

    @classmethod
    def from_maker(cls, maker: BaseAimsMaker) -> MolecularAimsMaker:
        """Adapt a standard AIMS Maker while retaining its base settings."""
        if isinstance(maker, cls):
            return maker
        if type(maker).make is not BaseAimsMaker.make:
            raise TypeError(
                "Maker has a custom make() method; cannot adapt automatically"
            )
        return cls(
            **{
                item.name: deepcopy(getattr(maker, item.name))
                for item in fields(BaseAimsMaker)
            }
        )

    @job
    def make(self, structure: Molecule) -> Response:
        """Write FHI-aims inputs, run the calculation, and parse its results."""
        write_options = dict(self.write_input_set_kwargs)
        if "prev_dir" in write_options or "directory" in write_options:
            raise ValueError("Molecular jobs manage their own working directory")
        properties = write_options.pop("properties", None)

        inputs = self.input_set_generator.get_input_set(
            structure, properties=properties
        )
        # Keep parameters.json: the standard AimsTaskDoc parser requires it.
        inputs.write_input(directory=".", **write_options)

        for filename, data in self.write_additional_data.items():
            output_filename = filename.replace(":", ".")
            dumpfn(data, output_filename)

        run_aims(**self.run_aims_kwargs)
        document = AimsTaskDoc.from_directory(
            Path.cwd(),
            **self.task_document_kwargs,
        )
        document.task_label = self.name

        # Compress only the FHI-aims files relevant to this molecular workflow.
        gzip_output_folder(
            directory=Path.cwd(),
            setting="atomate",
            files_list=[
                "geometry.in",
                "control.in",
                "aims.out",
                "geometry.in.next_step",
            ],
        )

        return Response(
            stop_children=should_stop_children(document, **self.stop_children_kwargs),
            output=document if self.store_output_data else None,
        )


@dataclass
class MolecularDisplacementMaker(MolecularAimsMaker):
    """Calculate molecular forces and dipole moments at fixed geometry."""

    name: str = "molecular IR static"
    input_set_generator: AimsInputGenerator = field(
        default_factory=lambda: StaticSetGenerator(
            user_params={
                "compute_forces": True,
                "output": ["dipole"],
            },
            use_structure_charge=True,
        )
    )


@job
def run_molecular_displacements(
    molecule: Molecule,
    displacement_maker: BaseAimsMaker,
    settings: MolecularIRSettings,
    molecule_id: str,
    correction_cycle: int = 0,
) -> Response:
    """Generate +/- Cartesian displacement calculations for every atom."""
    jobs = []
    outputs = []
    for atom in range(len(molecule)):
        for axis in range(3):
            for sign in (1, -1):
                displaced = deepcopy(molecule)
                vector = np.zeros(3)
                vector[axis] = sign * settings.displacement
                displaced.translate_sites([atom], vector)

                label = (
                    f"{molecule_id}.i_atom_{atom:02d}.i_coord_{axis}."
                    f"disp_{sign * settings.displacement:.4f}"
                )
                if correction_cycle:
                    label += f".cycle_{correction_cycle}"
                maker = deepcopy(displacement_maker)
                maker.name = label
                static = maker.make(displaced)
                static.metadata["molecular_ir"] = {
                    "kind": "displacement",
                    "molecule_id": molecule_id,
                    "cycle": correction_cycle,
                    "atom": atom,
                    "axis": axis,
                    "sign": sign,
                    "displacement": settings.displacement,
                }
                jobs.append(static)

                outputs.append(
                    {
                        "forces": static.output.output.forces,
                        "dipole": static.output.output.structure.properties["dipole"],
                        "uuid": static.uuid,
                        "dir_name": static.output.dir_name,
                    }
                )

    return Response(
        replace=Flow(
            jobs, output=outputs, name=f"{molecule_id} molecular displacements"
        )
    )


@job
def molecular_ir_job(
    molecule: Molecule,
    responses: list[dict[str, Any]],
    displacement_maker: BaseAimsMaker,
    relax_maker: BaseAimsMaker | None,
    spectrum_settings: SpectrumSettings,
    settings: MolecularIRSettings,
    molecule_id: str,
    correction_history: list[ImaginaryModeCorrection] | None = None,
) -> MolecularIRTaskDoc | Response:
    """Analyze the spectrum, correcting significant imaginary modes if needed."""
    history = list(correction_history or [])
    if len(responses) != 6 * len(molecule):
        raise ValueError(
            f"Expected {6 * len(molecule)} displacement responses, "
            f"received {len(responses)}"
        )

    document = MolecularIRTaskDoc.from_displacements(
        molecule=molecule,
        forces=[item["forces"] for item in responses],
        dipoles=[item["dipole"] for item in responses],
        displacement=settings.displacement,
        spectrum_settings=spectrum_settings,
        calculation_uuids=[item["uuid"] for item in responses],
        calculation_dirs=[item["dir_name"] for item in responses],
        imaginary_mode_threshold_cm_1=settings.imaginary_threshold_cm_1,
        imaginary_mode_correction_history=history,
    )

    frequencies = document.frequencies_cm_1
    if frequencies is None or len(frequencies) == 0:
        return document
    mode_index = int(np.argmin(frequencies))
    lowest_frequency = frequencies[mode_index]
    if lowest_frequency >= settings.imaginary_threshold_cm_1:
        return document
    if relax_maker is None or len(history) >= settings.max_imaginary_cycles:
        raise RuntimeError(
            "Significant imaginary vibrational modes remain; "
            "correction is unavailable or exhausted"
        )

    displaced = displace_imaginary_mode(
        document, settings.imaginary_displacement_angstrom, mode_index
    )
    cycle = len(history) + 1
    history.append(
        ImaginaryModeCorrection(
            cycle=cycle,
            frequency_cm_1=lowest_frequency,
            maximum_atomic_displacement_angstrom=settings.imaginary_displacement_angstrom,
        )
    )
    correction_maker = deepcopy(relax_maker)
    correction_maker.name = f"relaxation.cycle_{cycle:02d}"
    relaxation = correction_maker.make(displaced)
    relaxation.metadata["molecular_ir"] = {
        "kind": "relaxation",
        "molecule_id": molecule_id,
        "cycle": cycle,
    }
    relaxation.name = f"{molecule_id} {relax_maker.name} correction {cycle}"

    displacements = run_molecular_displacements(
        relaxation.output.structure,
        displacement_maker,
        settings,
        molecule_id,
        correction_cycle=cycle,
    )
    repeated = molecular_ir_job(
        molecule=relaxation.output.structure,
        responses=displacements.output,
        displacement_maker=displacement_maker,
        relax_maker=relax_maker,
        spectrum_settings=spectrum_settings,
        settings=settings,
        molecule_id=molecule_id,
        correction_history=history,
    )
    return Response(
        replace=Flow(
            [relaxation, displacements, repeated],
            output=repeated.output,
            name=f"{molecule_id} imaginary-mode correction {cycle}",
        )
    )
