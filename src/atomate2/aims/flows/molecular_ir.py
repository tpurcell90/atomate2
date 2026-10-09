"""Atomate2 flows for molecular conformer, FHI-aims, and IR calculations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from jobflow import Flow, Maker
from monty.serialization import loadfn
from pymatgen.io.aims.sets.core import RelaxSetGenerator
from rdkit import Chem as _Chem  # noqa: F401

from atomate2.aims.flows.core import RelaxMaker
from atomate2.aims.jobs.molecular_ir import (
    conformer_search_job,
    molecular_ir_job,
    save_optimization_job,
)
from atomate2.aims.schemas.molecular_ir import ConformerMetadata


def _default_relax_parameters() -> dict[str, Any]:
    """Return FHI-aims parameters used for molecular geometry optimization."""
    return {
        "xc": "b3lyp",
        "species_dir": "defaults_2020/tight",
        "spin": "none",
        "relativistic": "atomic_zora scalar",
        "charge": 0,
        "occupation_type": "gaussian 0.01",
        "sc_accuracy_rho": 1.0e-5,
        "sc_accuracy_eev": 1.0e-3,
        "sc_accuracy_etot": 1.0e-6,
        "sc_iter_limit": 200,
        "relax_geometry": "bfgs 5.0e-3",
    }


def _default_static_parameters() -> dict[str, Any]:
    """Return FHI-aims parameters used for displaced-geometry force jobs."""
    parameters = _default_relax_parameters()
    parameters.pop("relax_geometry")
    parameters.update(
        {
            "compute_forces": True,
            "final_forces_cleaned": True,
            "output": ["dipole"],
        }
    )
    return parameters


def _default_relax_maker() -> RelaxMaker:
    """Construct the standard Atomate2 FHI-aims relaxation Maker."""
    return RelaxMaker(
        name="FHI-aims molecular geometry optimization",
        input_set_generator=RelaxSetGenerator(
            user_params=_default_relax_parameters(),
        ),
    )


@dataclass
class MolecularIRMaker(Maker):
    """Build one complete SMILES-to-IR Atomate2 flow.

    The flow performs an RDKit conformer search, relaxes the lowest-energy
    conformer with the Atomate2 FHI-aims Maker, generates finite
    displacements, corrects imaginary modes when requested, and post-processes
    the spectrum into the target molar absorptivity.
    """

    name: str = "molecular IR workflow"
    relax_maker: RelaxMaker = field(default_factory=_default_relax_maker)
    get_vibrations_script: str | Path = "get_vibrations.py"
    results_directory: str | Path | None = None
    n_parallel_displacements: int = 1
    displacement_aims_cmd: str | None = None
    static_parameters: dict[str, Any] = field(
        default_factory=_default_static_parameters
    )
    n_conformers: int = 100
    allowed_elements: tuple[str, ...] = (
        "C",
        "H",
        # "B",
        # "N",
        # "O",
        # "F",
        # "Si",
        # "P",
        # "S",
        # "Cl",
        # "Br",
        # "I",
    )
    prune_rms_threshold: float = 0.25
    conformer_seeds: tuple[int, ...] = (61453, 72531, 83647)
    use_random_coordinates_fallback: bool = True
    imaginary_threshold_cm_1: float = -20.0
    imaginary_displacement_angstrom: float = 0.15
    max_imaginary_cycles: int = 3
    frequency_min_cm1: float = 800.0
    frequency_max_cm1: float = 1250.0
    linewidth_cm1: float = 10.0
    weight_fraction: float = 0.30
    polymer_density_g_ml: float = 1.70
    path_length_cm: float = 0.10
    wavenumber_step_cm1: float = 0.001

    def make(self, molecule_id: str, smiles: str) -> Flow:
        """Create a Jobflow Flow for one molecule."""
        results_dir = (
            Path(self.results_directory).expanduser().resolve()
            if self.results_directory
            else None
        )
        conformer_dir = results_dir / "conformer" if results_dir else None
        optimization_dir = results_dir / "aims_optimize" if results_dir else None
        ir_dir = results_dir / "ir" if results_dir else None
        dataset_dir = results_dir.parent if results_dir else None
        optimized_file = (
            optimization_dir / "optimized_molecule.json" if optimization_dir else None
        )
        metadata_file = (
            conformer_dir / "conformer_search.json" if conformer_dir else None
        )
        if (
            optimized_file
            and optimized_file.exists()
            and metadata_file
            and metadata_file.exists()
        ):
            molecule = loadfn(optimized_file)
            metadata = ConformerMetadata.model_validate_json(metadata_file.read_text())
            ir_job = molecular_ir_job(
                molecule=molecule,
                conformer=metadata,
                relax_maker=self.relax_maker,
                get_vibrations_script=str(self.get_vibrations_script),
                static_parameters=self.static_parameters,
                restart_directory=ir_dir,
                dataset_directory=dataset_dir,
                n_parallel_displacements=self.n_parallel_displacements,
                displacement_aims_cmd=self.displacement_aims_cmd,
                imaginary_threshold_cm_1=self.imaginary_threshold_cm_1,
                imaginary_displacement_angstrom=self.imaginary_displacement_angstrom,
                max_imaginary_cycles=self.max_imaginary_cycles,
                correction_cycle=0,
                weight_fraction=self.weight_fraction,
                polymer_density_g_ml=self.polymer_density_g_ml,
                path_length_cm=self.path_length_cm,
                min_wavenumber_cm_1=self.frequency_min_cm1,
                max_wavenumber_cm_1=self.frequency_max_cm1,
                wavenumber_step_cm_1=self.wavenumber_step_cm1,
                lorentzian_fwhm_cm_1=self.linewidth_cm1,
            )
            return Flow(
                [ir_job], output=ir_job.output, name=f"{self.name}: {molecule_id}"
            )
        conformer_job = conformer_search_job(
            molecule_id=molecule_id,
            smiles=smiles,
            n_conformers=self.n_conformers,
            allowed_elements=self.allowed_elements,
            prune_rms_angstrom=self.prune_rms_threshold,
            fallback_seeds=self.conformer_seeds,
            use_random_coordinates_fallback=self.use_random_coordinates_fallback,
            output_directory=conformer_dir,
        )

        relax_job = self.relax_maker.make(conformer_job.output.molecule)
        optimize_job = save_optimization_job(
            relax_job.output.output.structure,
            relax_job.output.dir_name,
            optimization_dir or Path.cwd() / "aims_optimize",
        )

        ir_job = molecular_ir_job(
            molecule=optimize_job.output,
            conformer=conformer_job.output.metadata,
            get_vibrations_script=str(self.get_vibrations_script),
            static_parameters=self.static_parameters,
            restart_directory=ir_dir,
            dataset_directory=dataset_dir,
            n_parallel_displacements=self.n_parallel_displacements,
            displacement_aims_cmd=self.displacement_aims_cmd,
            relax_maker=self.relax_maker,
            imaginary_threshold_cm_1=self.imaginary_threshold_cm_1,
            imaginary_displacement_angstrom=self.imaginary_displacement_angstrom,
            max_imaginary_cycles=self.max_imaginary_cycles,
            correction_cycle=0,
            weight_fraction=self.weight_fraction,
            polymer_density_g_ml=self.polymer_density_g_ml,
            path_length_cm=self.path_length_cm,
            min_wavenumber_cm_1=self.frequency_min_cm1,
            max_wavenumber_cm_1=self.frequency_max_cm1,
            wavenumber_step_cm_1=self.wavenumber_step_cm1,
            lorentzian_fwhm_cm_1=self.linewidth_cm1,
        )

        return Flow(
            jobs=[conformer_job, relax_job, optimize_job, ir_job],
            output=ir_job.output,
            name=f"{self.name}: {molecule_id}",
        )


@dataclass
class MolecularIRDatasetMaker(Maker):
    """Creates an IR Dataset."""

    name: str = "molecular IR dataset workflow"
    molecule_maker: MolecularIRMaker = field(default_factory=MolecularIRMaker)
    results_root: str | Path | None = None

    def make(self, records: list[dict[str, str]]) -> Flow:
        """Create one subflow per ``molecule_id``/``smiles`` record."""
        root = (
            Path(self.results_root).expanduser().resolve()
            if self.results_root
            else None
        )
        molecule_flows = []
        for record in records:
            molecule_id = record["molecule_id"]
            safe_id = "".join(
                character if character.isalnum() or character in "_.-" else "_"
                for character in molecule_id
            )
            maker = (
                replace(
                    self.molecule_maker, results_directory=root / safe_id[:2] / safe_id
                )
                if root
                else self.molecule_maker
            )
            molecule_flows.append(
                maker.make(molecule_id=molecule_id, smiles=record["smiles"])
            )
        return Flow(
            jobs=molecule_flows,
            output=[molecule_flow.output for molecule_flow in molecule_flows],
            name=self.name,
        )
