"""Define the molecular infrared workflow."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING

from jobflow import Flow, Maker
from pymatgen.core import Molecule
from pymatgen.io.aims.sets.core import RelaxSetGenerator

from atomate2.aims.jobs.core import RelaxMaker
from atomate2.aims.jobs.molecular_ir import (
    MolecularAimsMaker,
    MolecularDisplacementMaker,
    MolecularIRSettings,
    molecular_ir_job,
    run_molecular_displacements,
)
from atomate2.aims.schemas.molecular_ir import SpectrumSettings

if TYPE_CHECKING:
    from atomate2.aims.jobs.base import BaseAimsMaker


def _default_relax_maker() -> BaseAimsMaker:
    """Use TRM relaxation with a force threshold of 1e-3 eV/Å."""
    return RelaxMaker(
        input_set_generator=RelaxSetGenerator(
            relax_cell=False,
            method="trm",
            max_force=1e-3,
            use_structure_charge=True,
        )
    )


@dataclass
class MolecularIRMaker(Maker):
    """Optionally relax a molecule, then calculate its infrared spectrum."""

    name: str = "molecular IR"
    relax_maker: BaseAimsMaker | None = field(default_factory=_default_relax_maker)
    displacement_maker: BaseAimsMaker = field(
        default_factory=MolecularDisplacementMaker
    )
    spectrum_settings: SpectrumSettings = field(default_factory=SpectrumSettings)
    settings: MolecularIRSettings = field(default_factory=MolecularIRSettings)

    def make(self, molecule: Molecule, molecule_id: str = "molecule") -> Flow:
        """Create a molecular IR workflow."""
        if (
            not isinstance(molecule, Molecule)
            or not molecule.is_ordered
            or len(molecule) < 2
        ):
            raise ValueError("Expected an ordered Molecule with at least two atoms")
        if not isfinite(self.settings.displacement) or self.settings.displacement <= 0:
            raise ValueError("Displacement must be finite and positive")
        if (
            not isfinite(self.settings.imaginary_threshold_cm_1)
            or self.settings.imaginary_threshold_cm_1 >= 0
            or not isfinite(self.settings.imaginary_displacement_angstrom)
            or self.settings.imaginary_displacement_angstrom <= 0
            or self.settings.max_imaginary_cycles < 0
        ):
            raise ValueError("Invalid imaginary-mode correction settings")
        if not molecule_id:
            raise ValueError("molecule_id must be a nonempty string")

        displacement_maker = deepcopy(
            MolecularAimsMaker.from_maker(self.displacement_maker)
        )
        relax_maker = (
            deepcopy(MolecularAimsMaker.from_maker(self.relax_maker))
            if self.relax_maker is not None
            else None
        )
        for maker in (relax_maker, displacement_maker):
            if maker is not None:
                maker.input_set_generator.user_params.update(self.settings.user_params)

        if not displacement_maker.store_output_data:
            raise ValueError("Displacement Maker must have store_output_data=True")
        if relax_maker is not None and not relax_maker.store_output_data:
            raise ValueError("Relaxation Maker must have store_output_data=True")

        jobs = []
        optimized = molecule
        if relax_maker is not None:
            initial_maker = deepcopy(relax_maker)
            initial_maker.name = "relaxation.cycle_00"
            relaxation = initial_maker.make(molecule)
            relaxation.name = f"{molecule_id} {relaxation.name}"
            relaxation.metadata["molecular_ir"] = {
                "kind": "relaxation",
                "molecule_id": molecule_id,
                "cycle": 0,
            }
            jobs.append(relaxation)
            optimized = relaxation.output.structure

        displacements = run_molecular_displacements(
            optimized, displacement_maker, self.settings, molecule_id
        )
        analysis = molecular_ir_job(
            molecule=optimized,
            responses=displacements.output,
            displacement_maker=displacement_maker,
            relax_maker=relax_maker,
            spectrum_settings=self.spectrum_settings,
            settings=self.settings,
            molecule_id=molecule_id,
        )
        return Flow(
            [*jobs, displacements, analysis],
            output=analysis.output,
            name=f"{molecule_id} {self.name}",
        )
