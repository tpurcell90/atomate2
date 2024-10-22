"""A test suite for FHI-aims defect flows"""

import os

from jobflow import run_locally

# from numpy.testing import assert_allclose
from pymatgen.core import Structure

from atomate2.aims.flows.defect import (
    ConfigurationCoordinateMaker,
    FormationEnergyMaker,
)
from atomate2.common.schemas.defects import CCDDocument

cwd = os.getcwd()


def test_ccd_maker(si, tmp_path, mock_aims, test_dir, species_dir):
    ref_paths = {
        "Relaxation calculation q1": "ccd_si_vacancy/relax_q1",
        "Relaxation calculation q2": "ccd_si_vacancy/relax_q2",
    }

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    # generate flow
    si_defect = Structure.from_file(
        test_dir / "aims" / "ccd_si_vacancy" / "relax_q1" / "inputs" / "geometry.in.gz"
    )

    # generate flow
    ccd_maker = ConfigurationCoordinateMaker(distortions=(-0.2, -0.1, 0, 0.1, 0.2))
    assert ccd_maker.distortions == (-0.2, -0.1, 0, 0.1, 0.2)
    flow = ccd_maker.make(si_defect, charge_state1=0, charge_state2=1)

    # Run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    # validation on the outputs
    ccd_output = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(ccd_output, CCDDocument)

    # assert_allclose(
    #     elastic_output.elastic_tensor.ieee_format,
    #     [
    #         [147.279167, 56.2746603, 56.2746603, 0.0, 0.0, 0.0],
    #         [56.2746603, 147.279167, 56.2746603, 0.0, 0.0, 0.0],
    #         [56.2746603, 56.2746603, 147.279167, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 75.9240547, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 75.9240547, 0.0],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 75.9240547],
    #     ],
    #     atol=1e-6,
    # )
    # assert elastic_output.chemsys == "Si"


def test_fe_maker(si, tmp_path, mock_aims, test_dir, species_dir):
    from pymatgen.analysis.defects.generators import VacancyGenerator

    ref_paths = {
        "bulk relax": "defect-fe-si/bulk-relax",
        "Relaxation calculation v_Si-0 q=-1": "defect-fe-si/v_Si-0_q=-1",
        "Relaxation calculation v_Si-0 q=0": "defect-fe-si/v_Si-0_q=0",
        "Relaxation calculation v_Si-0 q=1": "defect-fe-si/v_Si-0_q=1",
    }

    # settings passed to fake_run_aims; adjust these to check for certain input settings
    fake_run_aims_kwargs = {}

    # automatically use fake FHI-aims
    mock_aims(ref_paths, fake_run_aims_kwargs)

    defects = list(VacancyGenerator().get_defects(structure=si, rm_species=["Si"]))
    # generate flow
    maker = FormationEnergyMaker(
        relax_radius="auto",
        perturb=0.1,
        collect_defect_entry_data=True,
        validate_charge=False,
    )
    flow = maker.make(
        defects[0],
        supercell_matrix=[[2, 2, 0], [2, -2, 0], [0, 0, 1]],
        defect_index=0,
    )

    # Run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    # validation on the outputs
    ccd_output = responses[flow.jobs[-1].uuid][1].output
    assert isinstance(ccd_output, CCDDocument)
