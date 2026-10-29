"""Test molecular IR workflows using saved FHI-aims reference calculations."""

import json

import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.core import Molecule
from pymatgen.io.aims.inputs import AimsGeometryIn

import atomate2.aims.jobs.base as aims_base_jobs
import atomate2.aims.jobs.molecular_ir as molecular_ir_jobs
from atomate2.aims.flows.molecular_ir import MolecularIRMaker
from atomate2.aims.jobs.molecular_ir import MolecularIRSettings
from atomate2.aims.schemas.molecular_ir import MolecularIRTaskDoc
from atomate2.aims.schemas.task import AimsTaskDoc


def _reference_paths(molecule_id, n_atoms, displacement):
    """Map generated job names to saved input/output reference directories."""
    root = f"molecular_ir/{molecule_id}/cycle_00"
    mapping = {f"{molecule_id} relaxation.cycle_00": f"{root}/relaxation"}
    for atom in range(n_atoms):
        for axis in range(3):
            for sign in (1, -1):
                label = (
                    f"{molecule_id}.i_atom_{atom:02d}.i_coord_{axis}."
                    f"disp_{sign * displacement:.4f}"
                )
                mapping[label] = f"{root}/displacements/{label}"
    return mapping


@pytest.mark.parametrize("molecule_id", ["ethene"])
def test_molecular_ir_workflow(
    mock_aims, monkeypatch, tmp_path, ref_path, species_dir, molecule_id
):
    """Run an IR workflow with saved molecular relaxation and displacement outputs."""
    root = ref_path / "molecular_ir" / molecule_id / "cycle_00"
    relaxation = root / "relaxation"
    geometry_file = relaxation / "inputs" / "geometry.in.gz"
    if not geometry_file.is_file():
        pytest.skip(f"Missing molecular IR reference geometry: {geometry_file}")

    molecule = AimsGeometryIn.from_file(geometry_file).structure
    assert isinstance(molecule, Molecule)

    settings = MolecularIRSettings(
        imaginary_threshold_cm_1=-1e6,
        user_params={"species_dir": str(species_dir)},
    )
    references = _reference_paths(molecule_id, len(molecule), settings.displacement)
    missing = [
        relative
        for relative in references.values()
        if not (ref_path / relative / "inputs" / "geometry.in.gz").is_file()
        or not (ref_path / relative / "outputs").is_dir()
        or not list((ref_path / relative / "outputs").glob("aims.out*"))
    ]
    if missing:
        pytest.skip(f"Missing {len(missing)} molecular IR reference calculations")

    parameters_file = relaxation / "inputs" / "parameters.json"
    if not parameters_file.is_file():
        pytest.fail(f"Missing reference input file: {parameters_file}")
    settings.user_params["xc"] = json.loads(parameters_file.read_text())["xc"]

    mock_aims(references)
    monkeypatch.setattr(molecular_ir_jobs, "run_aims", aims_base_jobs.run_aims)

    flow = MolecularIRMaker(settings=settings).make(molecule, molecule_id=molecule_id)
    monkeypatch.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)

    task_docs = [
        response.output
        for attempts in responses.values()
        for response in attempts.values()
        if isinstance(response.output, AimsTaskDoc)
    ]
    ir_docs = [
        response.output
        for attempts in responses.values()
        for response in attempts.values()
        if isinstance(response.output, MolecularIRTaskDoc)
    ]

    n_atoms = len(molecule)
    n_displacements = 6 * n_atoms
    n_modes = 3 * n_atoms
    assert len(task_docs) == n_displacements + 1
    assert len(ir_docs) == 1

    result = ir_docs[0]
    assert result.n_displacements == n_displacements
    assert len(result.calculation_uuids) == n_displacements
    assert len(result.frequencies_cm_1) == n_modes
    assert np.asarray(result.hessian_ev_angstrom_2).shape == (n_modes, n_modes)
    assert np.asarray(result.normal_modes).shape == (n_modes, n_atoms, 3)
    assert len(result.ir_intensities_km_mol) == n_modes
    assert np.all(np.isfinite(result.frequencies_cm_1))
    assert np.all(np.isfinite(result.ir_intensities_km_mol))
    assert np.all(np.asarray(result.ir_intensities_km_mol) >= 0)
    assert len(result.postprocessing.wavenumbers_cm_1) > 1
    assert all(
        isinstance(task.task_label, str) and task.task_label for task in task_docs
    )
