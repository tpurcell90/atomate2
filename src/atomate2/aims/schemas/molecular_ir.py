"""Molecular vibrational analysis and IR spectra."""

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from pymatgen.core import Molecule
from scipy.constants import Avogadro, atomic_mass, c, elementary_charge, epsilon_0
from scipy.integrate import simpson

from atomate2.aims.utils.units import omegaToTHz


class SpectrumSettings(BaseModel):
    """Lorentzian broadening and Beer-Lambert sample settings."""

    minimum_wavenumber_cm_1: float = Field(800.0, ge=0)
    maximum_wavenumber_cm_1: float = Field(1250.0, gt=0)
    step_cm_1: float = Field(0.1, gt=0)
    fwhm_cm_1: float = Field(10.0, gt=0)
    minimum_mode_cm_1: float = Field(10.0, ge=0)
    weight_fraction: float = Field(0.30, ge=0, le=1)
    density_g_ml: float = Field(1.70, gt=0)
    path_length_cm: float = Field(0.10, ge=0)


class IRPostProcessingDoc(BaseModel):
    """Broadened molecular IR spectrum."""

    settings: SpectrumSettings
    molecular_weight_g_mol: float
    concentration_mol_l: float
    wavenumbers_cm_1: list[float]
    epsilon_l_mol_1_cm_1: list[float]
    transmission_percent: list[float]
    average_epsilon_l_mol_1_cm_1: float
    average_transmission_percent: float

    @classmethod
    def from_modes(
        cls,
        molecule: Molecule,
        frequencies: np.ndarray,
        intensities: np.ndarray,
        settings: SpectrumSettings,
    ) -> Self:
        """Broaden integrated IR intensities in km/mol with normalized lines."""
        low = settings.minimum_wavenumber_cm_1
        high = settings.maximum_wavenumber_cm_1
        if high <= low or settings.step_cm_1 > high - low:
            raise ValueError(
                "Spectrum range must increase and contain at least two points"
            )
        count = int(np.ceil((high - low) / settings.step_cm_1)) + 1
        grid = np.linspace(low, high, count)
        epsilon = np.zeros_like(grid)
        half_width = settings.fwhm_cm_1 / 2
        for frequency, intensity in zip(frequencies, intensities, strict=True):
            if frequency <= settings.minimum_mode_cm_1:
                continue
            line = half_width / np.pi / ((grid - frequency) ** 2 + half_width**2)
            epsilon += intensity * 100 / np.log(10) * line
        weight = float(molecule.composition.weight)
        if weight <= 0:
            raise ValueError("Molecular weight must be positive")
        concentration = 1000 * settings.weight_fraction * settings.density_g_ml / weight
        transmission = 100 * 10 ** (-epsilon * concentration * settings.path_length_cm)
        return cls(
            settings=settings,
            molecular_weight_g_mol=weight,
            concentration_mol_l=concentration,
            wavenumbers_cm_1=grid.tolist(),
            epsilon_l_mol_1_cm_1=epsilon.tolist(),
            transmission_percent=transmission.tolist(),
            average_epsilon_l_mol_1_cm_1=float(simpson(epsilon, x=grid) / (high - low)),
            average_transmission_percent=float(
                simpson(transmission, x=grid) / (high - low)
            ),
        )


class ImaginaryModeCorrection(BaseModel):
    """Record one displacement along a negative vibrational mode."""

    cycle: int
    frequency_cm_1: float
    maximum_atomic_displacement_angstrom: float


class MolecularIRTaskDoc(BaseModel):
    """Single-molecule finite-difference vibrational task document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    molecule: Molecule
    frequencies_cm_1: list[float]
    normal_modes: list[list[list[float]]]
    hessian_ev_angstrom_2: list[list[float]]
    dipole_derivatives_e_amu_1_2: list[list[float]]
    ir_intensities_km_mol: list[float]
    displacement_angstrom: float
    n_displacements: int
    calculation_uuids: list[str] = Field(default_factory=list)
    calculation_dirs: list[str] = Field(default_factory=list)
    imaginary_mode_threshold_cm_1: float
    imaginary_mode_correction_history: list[ImaginaryModeCorrection] = Field(
        default_factory=list
    )
    postprocessing: IRPostProcessingDoc

    @classmethod
    def from_displacements(
        cls,
        molecule: Molecule,
        forces: list,
        dipoles: list,
        displacement: float,
        spectrum_settings: SpectrumSettings,
        **kwargs,
    ) -> Self:
        """Analyze ordered (+delta, -delta) pairs for each Cartesian coordinate."""
        size = 3 * len(molecule)
        force_array = np.asarray(forces, dtype=float)
        dipole_array = np.asarray(dipoles, dtype=float)
        if force_array.shape != (2 * size, len(molecule), 3):
            raise ValueError("Expected 6N force arrays with shape (N, 3)")
        if dipole_array.shape != (2 * size, 3):
            raise ValueError("Expected 6N dipole vectors with shape (3,)")
        if not np.all(np.isfinite(force_array)) or not np.all(
            np.isfinite(dipole_array)
        ):
            raise ValueError("Forces and dipoles must be finite")
        # Reference get_vibrations.py: +delta first, -delta second.
        # H_ij = -[F_i(R_j+delta) - F_i(R_j-delta)] / (2 delta).
        hessian = -(force_array[::2] - force_array[1::2]).reshape(size, size).T / (
            2 * displacement
        )
        hessian = (hessian + hessian.T) / 2

        masses = np.array([float(site.specie.atomic_mass) for site in molecule])
        inverse_sqrt_mass = 1 / np.sqrt(np.repeat(masses, 3))
        dynamical = hessian * np.outer(inverse_sqrt_mass, inverse_sqrt_mass)

        values, vectors = np.linalg.eigh(dynamical)
        frequencies = (
            np.sign(values) * np.sqrt(np.abs(values)) * omegaToTHz * 1e12 / (c * 100)
        )
        modes = (inverse_sqrt_mass[:, None] * vectors).T

        dipole_gradient = -(dipole_array[::2] - dipole_array[1::2]).T / (
            2 * displacement
        )
        derivatives = (dipole_gradient @ modes.T).T
        # SI conversion of N_A*pi/(3*epsilon_0*c²) * |d(mu)/d(Q)|² to km/mol.
        conversion = (
            Avogadro
            * elementary_charge**2
            / (12 * epsilon_0 * c**2 * atomic_mass * 1000)
        )
        intensities = conversion * np.sum(derivatives**2, axis=1)
        spectrum = IRPostProcessingDoc.from_modes(
            molecule, frequencies, intensities, spectrum_settings
        )
        return cls(
            molecule=molecule,
            frequencies_cm_1=frequencies.tolist(),
            normal_modes=modes.reshape(-1, len(molecule), 3).tolist(),
            hessian_ev_angstrom_2=hessian.tolist(),
            dipole_derivatives_e_amu_1_2=derivatives.tolist(),
            ir_intensities_km_mol=intensities.tolist(),
            displacement_angstrom=displacement,
            n_displacements=2 * size,
            postprocessing=spectrum,
            **kwargs,
        )
