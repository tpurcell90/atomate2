import os
from pathlib import Path

from jobflow import run_locally
from pymatgen.core.structure import Structure
from pymatgen.io.aims.sets.core import StaticSetGenerator

from atomate2.aims.flows.qha import QhaMaker
from atomate2.aims.jobs.core import RelaxMaker, StaticMaker

cwd = Path.cwd()


def test_qha(si: Structure, tmp_path, species_dir, mock_aims):
    # mapping from job name to directory containing test files
    ref_paths = {
        "EOS equilibrium relaxation": "Si_qha/EOS_equilibrum_relaxation",
        "static aims 1/1": "Si_qha/static_1_1",
        "SCF Calculation eos deformation 1": "Si_qha/static_eos_deformation_1",
    }

    # settings passed to fake_run_vasp; adjust these to check for certain INCAR settings
    fake_run_aims_kwargs = {}

    mock_aims(ref_paths, fake_run_aims_kwargs)

    parameters = {
        "species_dir": (species_dir / "light").as_posix(),
        "rlsy_symmetry": "all",
        "relativistic": "atomic_zora scalar",
    }

    parameters_phonon_disp = dict(compute_forces=True, **parameters)
    parameters_phonon_disp["rlsy_symmetry"] = None

    # generate job

    parameters_phonon_disp = dict(compute_forces=True, **parameters)

    maker = QhaMaker(
        number_of_frames=0,
        initial_relax_maker=RelaxMaker.full_relaxation(user_params=parameters),
        eos_relax_maker=RelaxMaker.full_relaxation(user_params=parameters),
        phonon_displacement_maker=StaticMaker(
            name="static aims",
            input_set_generator=StaticSetGenerator(
                user_params=parameters_phonon_disp,
                user_kpoints_settings={"density": 5.0, "even": True},
            ),
        ),
        phonon_static_maker=StaticMaker(
            input_set_generator=StaticSetGenerator(user_params=parameters)
        ),
        phonon_maker_kwargs={"min_length": 8, "born_maker": None},
        ignore_imaginary_modes=True,
        skip_analysis=True,
    )
    flow = maker.make(structure=si)

    # run the flow or job and ensure that it finished running successfully
    os.chdir(tmp_path)
    responses = run_locally(flow, create_folders=True, ensure_success=True)
    os.chdir(cwd)

    assert len(responses) == 9
