"""Schemas for molecular FHI-aims IR workflows."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pymatgen.core import Molecule


class ConformerMetadata(BaseModel):
    """RDKit conformer search."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_id: str
    smiles: str
    force_field: str
    best_conformer_id: int
    best_energy_kcal_mol: float
    final_optimization_status: int
    n_conformers_generated: int
    embedding_seed: int | None = None
    used_random_coordinates: bool = False


class ConformerSearchDoc(BaseModel):
    """Output produced by the conformer-search job."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule: Molecule
    metadata: ConformerMetadata


class ImaginaryModeCorrection(BaseModel):
    """Record of one distortion along an imaginary normal mode."""

    cycle: int
    mode_index_zero_based: int
    frequency_cm_1: float
    maximum_atomic_displacement_angstrom: float
    scale_factor: float
    source_directory: str


class IRPostProcessingDoc(BaseModel):
    """Summary of the broadened IR spectrum calculation."""

    molecular_weight_g_mol: float
    weight_fraction: float
    polymer_density_g_ml: float
    path_length_cm: float
    minimum_wavenumber_cm_1: float
    maximum_wavenumber_cm_1: float
    lorentzian_fwhm_cm_1: float
    average_epsilon_l_mol_1_cm_1: float
    average_transmission_percent: float
    spectrum_file: str
    summary_file: str


class MolecularIRTaskDoc(BaseModel):
    """Final task document for one molecular IR workflow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule_id: str
    smiles: str
    state: str = "successful"
    molecule: Molecule
    dir_name: str
    modes_file: str
    hessian_file: str
    eigenvectors_file: str
    dipole_gradient_file: str
    masses_file: str
    n_displacements: int
    imaginary_frequencies_cm_1: list[float] = Field(default_factory=list)
    imaginary_mode_threshold_cm_1: float
    imaginary_mode_correction_cycles: int = 0
    imaginary_mode_correction_history: list[ImaginaryModeCorrection] = Field(
        default_factory=list
    )
    conformer: ConformerMetadata
    postprocessing: IRPostProcessingDoc
    additional_fields: dict[str, Any] = Field(default_factory=dict)
