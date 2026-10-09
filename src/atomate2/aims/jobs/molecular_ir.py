"""Jobs for conformer search, molecular vibrations, and IR post-processing."""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jobflow import Flow, Maker, Response, job
from monty.json import MontyDecoder, MontyEncoder
from monty.serialization import dumpfn
from pymatgen.core import Molecule
from pymatgen.io.aims.sets.core import StaticSetGenerator
from pymatgen.io.xyz import XYZ
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.integrate import simpson

from atomate2.aims.run import run_aims
from atomate2.aims.schemas.molecular_ir import (
    ConformerMetadata,
    ConformerSearchDoc,
    ImaginaryModeCorrection,
    IRPostProcessingDoc,
    MolecularIRTaskDoc,
)

NUMBER_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?")


def _run_command(
    command: list[str] | str,
    *,
    cwd: Path,
    log_file: Path,
    shell: bool = False,
) -> None:
    with log_file.open("w") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=shell,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {command}"
        )


def _etkdg_parameters(
    seed: int, prune_rms: float, random_coords: bool
) -> Chem.rdDistGeom.EmbedParameters:
    """Return configured RDKit ETKDGv3 parameters."""
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.pruneRmsThresh = float(prune_rms)
    parameters.useSmallRingTorsions = True
    parameters.useMacrocycleTorsions = True
    parameters.useRandomCoords = bool(random_coords)
    return parameters


def _coordinates(molecule: Chem.Mol, conformer_id: int) -> tuple[list[str], np.ndarray]:
    """Return symbols and Cartesian coordinates for one RDKit conformer."""
    conformer = molecule.GetConformer(int(conformer_id))
    symbols: list[str] = []
    coordinates: list[list[float]] = []
    for atom in molecule.GetAtoms():
        position = conformer.GetAtomPosition(atom.GetIdx())
        symbols.append(atom.GetSymbol())
        coordinates.append([position.x, position.y, position.z])
    return symbols, np.asarray(coordinates, dtype=float)


def _write_conformer_trajectory(
    molecule: Chem.Mol,
    results: dict[int, tuple[int, float]],
    filename: Path,
) -> None:
    """Write all optimized conformers using Pymatgen's Molecule."""
    frames: list[Molecule] = []
    energy_records: list[dict[str, int | float]] = []
    for conformer_id, (not_converged, energy) in results.items():
        symbols, coordinates = _coordinates(molecule, conformer_id)
        frames.append(Molecule(symbols, coordinates))
        energy_records.append(
            {
                "conformer_id": conformer_id,
                "energy_kcal_mol": energy,
                "not_converged": not_converged,
            }
        )

    XYZ(frames, coord_precision=8).write_file(str(filename))
    pd.DataFrame(energy_records).to_csv(
        filename.with_name(f"{filename.stem}_energies.csv"),
        index=False,
    )


@job(output_schema=ConformerSearchDoc)
def conformer_search_job(
    smiles: str,
    molecule_id: str,
    *,
    allowed_elements: tuple[str, ...] = ("C", "H"),
    n_conformers: int = 100,
    max_iterations: int = 1000,
    prune_rms_angstrom: float = 0.25,
    fallback_seeds: tuple[int, ...] = (1234, 2024, 777, 9999, 314159),
    allow_uff_fallback: bool = True,
    use_random_coordinates_fallback: bool = True,
    save_all_conformers: bool = True,
    output_directory: str | Path | None = None,
) -> ConformerSearchDoc:
    """Generate ETKDG conformers and select the lowest-energy optimized one."""
    rdkit_molecule = Chem.MolFromSmiles(smiles)
    if rdkit_molecule is None:
        raise ValueError(f"RDKit could not parse {molecule_id}: {smiles}")
    rdkit_molecule = Chem.AddHs(rdkit_molecule)

    symbols = {atom.GetSymbol() for atom in rdkit_molecule.GetAtoms()}
    unsupported = sorted(symbols - set(allowed_elements))
    if unsupported:
        raise ValueError(f"Unsupported elements for {molecule_id}: {unsupported}")

    mmff = AllChem.MMFFGetMoleculeProperties(rdkit_molecule, mmffVariant="MMFF94")
    if mmff is not None:
        force_field = "MMFF94"
    elif allow_uff_fallback:
        force_field = "UFF"
    else:
        raise RuntimeError(f"MMFF94 parameters unavailable for {molecule_id}")

    conformer_ids: list[int] = []
    seed_used: int | None = None
    used_random_coordinates = False
    coordinate_modes = [False, True] if use_random_coordinates_fallback else [False]
    for random_coordinates in coordinate_modes:
        for seed in fallback_seeds:
            parameters = _etkdg_parameters(seed, prune_rms_angstrom, random_coordinates)
            conformer_ids = list(
                AllChem.EmbedMultipleConfs(
                    rdkit_molecule,
                    numConfs=int(n_conformers),
                    params=parameters,
                )
            )
            if conformer_ids:
                seed_used = seed
                used_random_coordinates = random_coordinates
                break
        if conformer_ids:
            break
    if not conformer_ids:
        raise RuntimeError(f"RDKit embedding failed for {molecule_id}")

    if force_field == "MMFF94":
        raw_results = AllChem.MMFFOptimizeMoleculeConfs(
            rdkit_molecule,
            mmffVariant="MMFF94",
            maxIters=max_iterations,
        )
    else:
        raw_results = AllChem.UFFOptimizeMoleculeConfs(
            rdkit_molecule,
            maxIters=max_iterations,
        )
    results = {
        int(conformer_id): (int(flag), float(energy))
        for conformer_id, (flag, energy) in zip(conformer_ids, raw_results, strict=True)
    }
    converged = {
        conformer_id: energy
        for conformer_id, (flag, energy) in results.items()
        if flag == 0
    }
    candidates = converged or {
        conformer_id: energy for conformer_id, (_, energy) in results.items()
    }
    best_id = min(candidates, key=candidates.get)
    best_energy = results[best_id][1]

    if force_field == "MMFF94":
        final_status = AllChem.MMFFOptimizeMolecule(
            rdkit_molecule,
            mmffVariant="MMFF94",
            confId=best_id,
            maxIters=max_iterations,
        )
    else:
        final_status = AllChem.UFFOptimizeMolecule(
            rdkit_molecule,
            confId=best_id,
            maxIters=max_iterations,
        )

    output_dir = (
        Path(output_directory).expanduser().resolve()
        if output_directory
        else Path.cwd()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    species, cartesian_coordinates = _coordinates(rdkit_molecule, best_id)
    molecule = Molecule(species, cartesian_coordinates)
    XYZ(molecule, coord_precision=12).write_file(
        str(output_dir / "rdkit_lowest_energy_conformer.xyz")
    )
    if save_all_conformers:
        _write_conformer_trajectory(
            rdkit_molecule,
            results,
            output_dir / "rdkit_all_optimized_conformers.xyz",
        )

    metadata = ConformerMetadata(
        molecule_id=molecule_id,
        smiles=smiles,
        force_field=force_field,
        best_conformer_id=best_id,
        best_energy_kcal_mol=best_energy,
        final_optimization_status=int(final_status),
        n_conformers_generated=len(conformer_ids),
        embedding_seed=seed_used,
        used_random_coordinates=used_random_coordinates,
    )
    (output_dir / "conformer_search.json").write_text(
        metadata.model_dump_json(indent=2) + "\n"
    )
    return ConformerSearchDoc(molecule=molecule, metadata=metadata)


@job
def save_optimization_job(
    molecule: Molecule, source_directory: str, output_directory: str | Path
) -> Molecule:
    """Save the completed FHI-aims optimization."""
    source_text = str(source_directory)
    remote_path = re.match(r"^[^/:]+:(/.*)$", source_text)
    source = Path(remote_path.group(1) if remote_path else source_text)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    dumpfn(molecule, output / "optimized_molecule.json")
    for pattern in ("geometry.in", "geometry.in.next_step", "control.in"):
        for filename in source.glob(pattern):
            shutil.copy2(filename, output / filename.name)
    aims_output = source / "aims.out"
    compressed_aims_output = source / "aims.out.gz"
    if compressed_aims_output.exists():
        shutil.copy2(compressed_aims_output, output / "aims.out.gz")
    elif aims_output.exists():
        with (
            aims_output.open("rb") as source_handle,
            gzip.open(output / "aims.out.gz", "wb") as output_handle,
        ):
            shutil.copyfileobj(source_handle, output_handle)
    else:
        raise FileNotFoundError(f"No aims.out or aims.out.gz found in {source}")
    return molecule


def _read_modes(modes_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read frequency and IR-intensity columns from get_vibrations.py output."""
    frequencies: list[float] = []
    intensities: list[float] = []
    for line in modes_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        numbers = [
            float(value.replace("D", "E").replace("d", "e"))
            for value in NUMBER_PATTERN.findall(stripped)
        ]
        if len(numbers) >= 4:
            frequencies.append(numbers[1])
            intensities.append(numbers[-1])
    if not frequencies:
        raise RuntimeError(f"No vibrational modes parsed from {modes_file}")
    return np.asarray(frequencies), np.asarray(intensities)


def _displace_imaginary_mode(
    molecule: Molecule,
    frequencies: np.ndarray,
    eigenvectors_file: Path,
    masses_file: Path,
    maximum_displacement: float,
    eigenvectors_are_columns: bool,
) -> tuple[Molecule, int, float, float]:
    """Return a molecule displaced along its most-negative normal mode."""
    n_cartesian = 3 * len(molecule)
    eigenvectors = np.loadtxt(eigenvectors_file, comments="#")
    if eigenvectors.shape != (n_cartesian, n_cartesian):
        raise RuntimeError(
            f"Expected eigenvector matrix {(n_cartesian, n_cartesian)}, "
            f"found {eigenvectors.shape}"
        )
    mode_index = int(np.argmin(frequencies))
    vector = (
        eigenvectors[:, mode_index]
        if eigenvectors_are_columns
        else eigenvectors[mode_index, :]
    )
    masses = np.loadtxt(masses_file, comments="#").reshape(-1)
    if masses.size == len(molecule):
        masses = np.repeat(masses, 3)
    if masses.size != n_cartesian:
        raise RuntimeError(f"Expected {len(molecule)} or {n_cartesian} masses")
    cartesian_mode = (vector / np.sqrt(masses)).reshape(len(molecule), 3)
    largest_displacement = float(np.max(np.linalg.norm(cartesian_mode, axis=1)))
    if largest_displacement <= 0:
        raise RuntimeError("Selected imaginary-mode eigenvector is zero")
    scale = maximum_displacement / largest_displacement
    displaced_coordinates = molecule.cart_coords + scale * cartesian_mode
    displaced = Molecule(
        [site.specie for site in molecule],
        displaced_coordinates,
        charge=molecule.charge,
        spin_multiplicity=molecule.spin_multiplicity,
    )
    return displaced, mode_index, float(frequencies[mode_index]), float(scale)


def _postprocess_ir(
    molecule: Molecule,
    modes_file: Path,
    *,
    output_dir: Path,
    summary_file: Path,
    weight_fraction: float,
    polymer_density_g_ml: float,
    path_length_cm: float,
    min_wavenumber: float,
    max_wavenumber: float,
    step: float,
    fwhm: float,
) -> IRPostProcessingDoc:
    """Broaden modes and calculate average epsilon and % transmission."""
    frequencies, aims_intensities = _read_modes(modes_file)
    wavenumbers = np.arange(min_wavenumber, max_wavenumber + 1.0e-10, step)
    epsilon = np.zeros_like(wavenumbers)
    half_width = 0.5 * fwhm
    for frequency, aims_intensity in zip(frequencies, aims_intensities, strict=True):
        if frequency < 10.0:
            continue
        intensity_km_mol = 42.255 * aims_intensity
        prefactor = intensity_km_mol * 100.0 / np.log(10.0)
        epsilon += (
            prefactor
            * half_width
            / np.pi
            * (wavenumbers / frequency)
            / ((wavenumbers - frequency) ** 2 + half_width**2)
        )

    molecular_weight = float(molecule.composition.weight)
    concentration = 1000.0 * weight_fraction * polymer_density_g_ml / molecular_weight
    transmission = 10.0 ** (2.0 - epsilon * concentration * path_length_cm)
    width = max_wavenumber - min_wavenumber
    average_epsilon = float(simpson(y=epsilon, x=wavenumbers) / width)
    average_transmission = float(simpson(y=transmission, x=wavenumbers) / width)

    spectrum_file = output_dir / "spectra_1000um.csv.gz"
    pd.DataFrame(
        {
            "wavenumber_cm-1": wavenumbers,
            "epsilon_L_mol-1_cm-1": epsilon,
            "transmission_percent": transmission,
        }
    ).to_csv(spectrum_file, index=False, compression="gzip")
    return IRPostProcessingDoc(
        molecular_weight_g_mol=molecular_weight,
        weight_fraction=weight_fraction,
        polymer_density_g_ml=polymer_density_g_ml,
        path_length_cm=path_length_cm,
        minimum_wavenumber_cm_1=min_wavenumber,
        maximum_wavenumber_cm_1=max_wavenumber,
        lorentzian_fwhm_cm_1=fwhm,
        average_epsilon_l_mol_1_cm_1=average_epsilon,
        average_transmission_percent=average_transmission,
        spectrum_file=_atomate2_runs_path(spectrum_file),
        summary_file=_atomate2_runs_path(summary_file),
    )


def _atomate2_runs_path(path: Path) -> str:
    """Return a result path."""
    resolved = path.resolve()
    indices = [
        index for index, part in enumerate(resolved.parts) if part == "atomate2_runs"
    ]
    if not indices:
        raise ValueError(f"Result is not inside atomate2_runs: {resolved}")
    return Path(*resolved.parts[indices[-1] :]).as_posix()


def _update_dataset_csv(
    summary_file: Path,
    molecule_id: str,
    smiles: str,
    postprocessing: IRPostProcessingDoc,
) -> None:
    """Insert or replace one molecule in the results CSV."""
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "molecule_id": molecule_id,
        "smiles": smiles,
        "molecular_weight_g_mol": postprocessing.molecular_weight_g_mol,
        "weight_fraction": postprocessing.weight_fraction,
        "polymer_density_g_ml": postprocessing.polymer_density_g_ml,
        "path_length_cm": postprocessing.path_length_cm,
        "minimum_wavenumber_cm_1": postprocessing.minimum_wavenumber_cm_1,
        "maximum_wavenumber_cm_1": postprocessing.maximum_wavenumber_cm_1,
        "lorentzian_fwhm_cm_1": postprocessing.lorentzian_fwhm_cm_1,
        "average_epsilon_l_mol_1_cm_1": postprocessing.average_epsilon_l_mol_1_cm_1,
        "average_transmission_percent": postprocessing.average_transmission_percent,
        "spectrum_file": postprocessing.spectrum_file,
    }
    lock_file = summary_file.with_suffix(summary_file.suffix + ".lock")
    with lock_file.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        existing = (
            pd.read_csv(summary_file) if summary_file.exists() else pd.DataFrame()
        )
        if "molecule_id" in existing.columns:
            existing = existing[existing["molecule_id"].astype(str) != str(molecule_id)]
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        temporary = summary_file.with_suffix(summary_file.suffix + ".tmp")
        updated.to_csv(temporary, index=False)
        os.replace(temporary, summary_file)
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


def _archive_displacements(ir_directory: Path, safe_id: str) -> Path:
    """Archive all displacement directories into one tar.gz file."""
    archive = ir_directory / "displacement_calculations.tar.gz"
    displacement_directories = sorted(
        path
        for path in ir_directory.glob(f"cycle_*/{safe_id}.i_atom_*.i_coord_*.disp_*")
        if path.is_dir()
    )
    if not archive.exists() and displacement_directories:
        temporary = archive.with_suffix(archive.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        with tarfile.open(temporary, "w:gz") as handle:
            for directory in displacement_directories:
                handle.add(directory, arcname=directory.relative_to(ir_directory))
        os.replace(temporary, archive)
    if archive.exists():
        with tarfile.open(archive, "r:gz") as handle:
            handle.getmembers()
        for directory in displacement_directories:
            shutil.rmtree(directory)
    return archive


def _aims_complete(directory: Path) -> bool:
    """Return whether a displacement finished normally."""
    output = directory / "aims.out"
    return output.is_file() and "Have a nice day." in output.read_text(errors="ignore")


def _run_displacement(directory: str | Path, aims_cmd: str | None) -> str:
    """Run one displacement."""
    directory = Path(directory)
    if _aims_complete(directory):
        return str(directory)
    previous_directory = Path.cwd()
    try:
        os.chdir(directory)
        run_aims(aims_cmd=aims_cmd)
    finally:
        os.chdir(previous_directory)
    if not _aims_complete(directory):
        raise RuntimeError(f"FHI-aims did not finish normally in {directory}")
    return str(directory)


@job(output_schema=MolecularIRTaskDoc)
def molecular_ir_job(
    molecule: Molecule,
    conformer: ConformerMetadata,
    relax_maker: Maker,
    *,
    get_vibrations_script: str,
    static_parameters: dict[str, Any],
    restart_directory: str | Path | None = None,
    dataset_directory: str | Path | None = None,
    n_parallel_displacements: int = 1,
    displacement_aims_cmd: str | None = None,
    displacement_angstrom: float = 0.0025,
    imaginary_threshold_cm_1: float = -20.0,
    imaginary_displacement_angstrom: float = 0.10,
    max_imaginary_cycles: int = 3,
    correction_cycle: int = 0,
    correction_history: list[ImaginaryModeCorrection] | None = None,
    eigenvectors_are_columns: bool = True,
    weight_fraction: float = 0.30,
    polymer_density_g_ml: float = 1.70,
    path_length_cm: float = 0.10,
    min_wavenumber_cm_1: float = 800.0,
    max_wavenumber_cm_1: float = 1250.0,
    wavenumber_step_cm_1: float = 0.001,
    lorentzian_fwhm_cm_1: float = 10.0,
) -> MolecularIRTaskDoc | Response:
    """Run finite-difference IR and post-process."""
    molecule_id = conformer.molecule_id
    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", molecule_id)
    workdir = (
        Path(restart_directory).expanduser().resolve() / f"cycle_{correction_cycle}"
        if restart_directory
        else Path.cwd()
    )
    workdir.mkdir(parents=True, exist_ok=True)
    final_document_file = (
        Path(restart_directory).expanduser().resolve() / "molecular_ir_task.json"
        if restart_directory
        else workdir / "molecular_ir_task.json"
    )
    if final_document_file.exists():
        document = MolecularIRTaskDoc.model_validate(
            json.loads(final_document_file.read_text(), cls=MontyDecoder)
        )
        if restart_directory:
            _archive_displacements(
                Path(restart_directory).expanduser().resolve(), safe_id
            )
        return document
    history = list(correction_history or [])
    restart_molecule_file = workdir / "restart_molecule.json"
    if restart_molecule_file.exists():
        restart_molecule = json.loads(
            restart_molecule_file.read_text(), cls=MontyDecoder
        )
        same_species = [str(specie) for specie in restart_molecule.species] == [
            str(specie) for specie in molecule.species
        ]
        if not same_species or not np.allclose(
            restart_molecule.cart_coords, molecule.cart_coords, atol=1.0e-4, rtol=0.0
        ):
            raise RuntimeError(
                "Restart geometry differs from the current geometry. "
                "Remove or rename {workdir} to start a new calculation."
            )
    else:
        restart_molecule_file.write_text(
            json.dumps(molecule, cls=MontyEncoder, indent=2) + "\n"
        )

    input_set = StaticSetGenerator(user_params=static_parameters).get_input_set(
        molecule
    )
    input_set.write_input(workdir)
    (workdir / "parameters.json").unlink(
        missing_ok=True
    )  ## Does not write parameters.json
    control_text = (workdir / "control.in").read_text()
    if not re.search(r"^\s*output\s+dipole\b", control_text, flags=re.MULTILINE):
        raise RuntimeError(
            "The displacement control.in does not contain 'output dipole'. "
            "IR intensities require a dipole moment for every displaced geometry."
        )
    shutil.copy2(workdir / "geometry.in", workdir / "geometry.in.next_step")

    source_script = Path(get_vibrations_script).expanduser().resolve()
    if not source_script.exists():
        raise FileNotFoundError(source_script)
    # shutil.copy2(source_script, workdir / "get_vibrations.py")
    displacement_directories = sorted(
        directory
        for directory in workdir.glob(f"{safe_id}.i_atom_*.i_coord_*.disp_*")
        if directory.is_dir()
    )
    expected_displacements = 6 * len(molecule)
    if len(displacement_directories) != expected_displacements:
        _run_command(
            [
                sys.executable,
                str(source_script),
                safe_id,
                "prepare",
                "-d",
                str(displacement_angstrom),
            ],
            cwd=workdir,
            log_file=workdir / "get_vibrations_prepare.log",
        )
        displacement_directories = sorted(
            directory
            for directory in workdir.glob(f"{safe_id}.i_atom_*.i_coord_*.disp_*")
            if directory.is_dir()
        )
    if len(displacement_directories) != expected_displacements:
        raise RuntimeError(
            f"Expected {expected_displacements} displacement directories "
            f"for {molecule_id}, found {len(displacement_directories)}"
        )

    pending = [
        directory
        for directory in displacement_directories
        if not _aims_complete(directory)
    ]
    if n_parallel_displacements < 1:
        raise ValueError("n_parallel_displacements must be at least 1")
    if n_parallel_displacements == 1:
        for _index, directory in enumerate(pending, start=1):
            _run_displacement(directory, displacement_aims_cmd)
    else:
        with ProcessPoolExecutor(max_workers=n_parallel_displacements) as executor:
            futures = {
                executor.submit(
                    _run_displacement, directory, displacement_aims_cmd
                ): directory
                for directory in pending
            }
            for _index, future in enumerate(as_completed(futures), start=1):
                directory = futures[future]
                future.result()
    _run_command(
        [sys.executable, str(source_script), "--IR", safe_id, "analysis"],
        cwd=workdir,
        log_file=workdir / "get_vibrations_analysis.log",
    )

    modes_file = workdir / f"modes.{safe_id}.dat"
    frequencies, _ = _read_modes(modes_file)
    imaginary = [float(value) for value in frequencies if value < 0]
    meaningful = [value for value in imaginary if value < imaginary_threshold_cm_1]

    if meaningful:
        if correction_cycle >= max_imaginary_cycles:
            raise RuntimeError(
                f"Imaginary modes remain after {max_imaginary_cycles} "
                f"cycles: {meaningful}"
            )
        displaced, mode_index, frequency, scale = _displace_imaginary_mode(
            molecule,
            frequencies,
            workdir / f"eigen_vectors.{safe_id}.dat",
            workdir / f"masses.{safe_id}.dat",
            imaginary_displacement_angstrom,
            eigenvectors_are_columns,
        )
        correction = ImaginaryModeCorrection(
            cycle=correction_cycle + 1,
            mode_index_zero_based=mode_index,
            frequency_cm_1=frequency,
            maximum_atomic_displacement_angstrom=imaginary_displacement_angstrom,
            scale_factor=scale,
            source_directory=str(workdir),
        )
        new_history = [*history, correction]
        corrected_relaxation = relax_maker.make(displaced)
        corrected_relaxation.name = (
            f"{safe_id}: imaginary-mode relaxation {correction_cycle + 1}"
        )
        repeated_ir = molecular_ir_job(
            corrected_relaxation.output.output.structure,
            conformer,
            relax_maker,
            get_vibrations_script=get_vibrations_script,
            static_parameters=static_parameters,
            restart_directory=restart_directory,
            dataset_directory=dataset_directory,
            n_parallel_displacements=n_parallel_displacements,
            displacement_aims_cmd=displacement_aims_cmd,
            displacement_angstrom=displacement_angstrom,
            imaginary_threshold_cm_1=imaginary_threshold_cm_1,
            imaginary_displacement_angstrom=imaginary_displacement_angstrom,
            max_imaginary_cycles=max_imaginary_cycles,
            correction_cycle=correction_cycle + 1,
            correction_history=new_history,
            eigenvectors_are_columns=eigenvectors_are_columns,
            weight_fraction=weight_fraction,
            polymer_density_g_ml=polymer_density_g_ml,
            path_length_cm=path_length_cm,
            min_wavenumber_cm_1=min_wavenumber_cm_1,
            max_wavenumber_cm_1=max_wavenumber_cm_1,
            wavenumber_step_cm_1=wavenumber_step_cm_1,
            lorentzian_fwhm_cm_1=lorentzian_fwhm_cm_1,
        )
        repeated_ir.name = f"{safe_id}: corrected IR {correction_cycle + 1}"
        replacement = Flow(
            [corrected_relaxation, repeated_ir],
            output=repeated_ir.output,
            name=f"{safe_id}: imaginary-mode correction",
        )
        return Response(replace=replacement)

    summary_file = (
        Path(dataset_directory).expanduser().resolve() / "all_results_1000um.csv"
        if dataset_directory
        else workdir / "all_results_1000um.csv"
    )
    postprocessing = _postprocess_ir(
        molecule,
        modes_file,
        output_dir=workdir,
        summary_file=summary_file,
        weight_fraction=weight_fraction,
        polymer_density_g_ml=polymer_density_g_ml,
        path_length_cm=path_length_cm,
        min_wavenumber=min_wavenumber_cm_1,
        max_wavenumber=max_wavenumber_cm_1,
        step=wavenumber_step_cm_1,
        fwhm=lorentzian_fwhm_cm_1,
    )
    _update_dataset_csv(summary_file, molecule_id, conformer.smiles, postprocessing)
    document = MolecularIRTaskDoc(
        molecule_id=molecule_id,
        smiles=conformer.smiles,
        molecule=molecule,
        dir_name=str(workdir),
        modes_file=str(modes_file),
        hessian_file=str(workdir / f"hessian.{safe_id}.dat"),
        eigenvectors_file=str(workdir / f"eigen_vectors.{safe_id}.dat"),
        dipole_gradient_file=str(workdir / f"grad_dipole.{safe_id}.dat"),
        masses_file=str(workdir / f"masses.{safe_id}.dat"),
        n_displacements=len(displacement_directories),
        imaginary_frequencies_cm_1=imaginary,
        imaginary_mode_threshold_cm_1=imaginary_threshold_cm_1,
        imaginary_mode_correction_cycles=correction_cycle,
        imaginary_mode_correction_history=history,
        conformer=conformer,
        postprocessing=postprocessing,
    )
    final_document_file.write_text(
        json.dumps(document.model_dump(), cls=MontyEncoder, indent=2) + "\n"
    )
    if restart_directory:
        _archive_displacements(Path(restart_directory).expanduser().resolve(), safe_id)
    return document
