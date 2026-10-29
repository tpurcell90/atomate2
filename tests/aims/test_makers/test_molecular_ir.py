"""Test molecular FHI-aims makers against saved calculation outputs."""

import numpy as np
import pytest
from jobflow import run_locally
from pymatgen.core import Molecule
from pymatgen.io.aims.inputs import AimsGeometryIn

import atomate2.aims.jobs.base as aims_base_jobs
import atomate2.aims.jobs.molecular_ir as molecular_ir_jobs
from atomate2.aims.jobs.molecular_ir import MolecularDisplacementMaker
from atomate2.aims.schemas.task import AimsTaskDoc


@pytest.mark.parametrize("molecule_id", ["ethene"])
@pytest.mark.parametrize("sign", [1, -1])
def test_molecular_displacement_maker(
    mock_aims, monkeypatch, tmp_path, ref_path, species_dir, molecule_id, sign
):
    """Parse a saved Cartesian displacement calculation for any reference molecule."""
    displacement = 0.0025
    label = f"{molecule_id}.i_atom_00.i_coord_0.disp_{sign * displacement:.4f}"
    relative = f"molecular_ir/{molecule_id}/cycle_00/displacements/{label}"
    reference = ref_path / relative
    geometry_file = reference / "inputs" / "geometry.in.gz"
    if not geometry_file.is_file() or not (reference / "outputs").is_dir():
        pytest.skip(f"Missing molecular IR reference calculation: {relative}")

    molecule = AimsGeometryIn.from_file(geometry_file).structure
    assert isinstance(molecule, Molecule)

    maker = MolecularDisplacementMaker(name=label)
    maker.input_set_generator.user_params["species_dir"] = str(species_dir)
    mock_aims({label: relative})
    monkeypatch.setattr(molecular_ir_jobs, "run_aims", aims_base_jobs.run_aims)

    monkeypatch.chdir(tmp_path)
    responses = run_locally(
        maker.make(molecule), create_folders=True, ensure_success=True
    )
    task_docs = [
        response.output
        for attempts in responses.values()
        for response in attempts.values()
        if isinstance(response.output, AimsTaskDoc)
    ]
    assert len(task_docs) == 1

    task = task_docs[0]
    assert isinstance(task.task_label, str)
    assert task.task_label
    assert isinstance(task.output.structure, Molecule)
    forces = np.asarray(task.output.forces, dtype=float)
    dipole = np.asarray(task.output.structure.properties["dipole"], dtype=float)
    assert forces.shape == (len(molecule), 3)
    assert dipole.shape == (3,)
    assert np.all(np.isfinite(forces))
    assert np.all(np.isfinite(dipole))
