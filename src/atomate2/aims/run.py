"""An FHI-aims jobflow runner."""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import subprocess
from glob import glob
from os.path import expandvars
from pathlib import Path
from typing import TYPE_CHECKING

from ase.calculators.aims import Aims, AimsProfile
from ase.calculators.socketio import SocketIOCalculator
from monty.json import MontyDecoder
from pymatgen.io.ase import AseAtomsAdaptor

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from pymatgen.core import Molecule, Structure

    from atomate2.aims.schemas.task import AimsTaskDoc
logger = logging.getLogger(__name__)


def run_aims(
    aims_cmd: str = None,
) -> None:
    """
    Run FHI-aims.

    Parameters
    ----------
    aims_cmd : str
        The command used to run FHI-aims (defaults to SETTINGS.AIMS_CMD).
    """
    if aims_cmd is None:
        aims_cmd = SETTINGS.AIMS_CMD

    aims_cmd = expandvars(aims_cmd)

    logger.info(f"Running command: {aims_cmd}")
    return_code = subprocess.call(["/bin/bash", "-c", aims_cmd], env=os.environ)
    logger.info(f"{aims_cmd} finished running with return code: {return_code}")


def should_stop_children(
    task_document: AimsTaskDoc,
    handle_unsuccessful: bool | str = True,
) -> bool:
    """
    Decide whether child jobs should continue.

    Parameters
    ----------
    task_document : .TaskDoc
        An FHI-aims task document.
    handle_unsuccessful : bool or str
        This is a three-way toggle on what to do if your job looks OK, but is actually
        not converged (either electronic or ionic):

        - `True`: Mark job as completed, but stop children.
        - `False`: Do nothing, continue with workflow as normal.
        - `"error"`: Throw an error.

    Returns
    -------
    bool
        Whether to stop child jobs.
    """
    if task_document.state == "successful":
        return False

    if isinstance(handle_unsuccessful, bool):
        return handle_unsuccessful

    if handle_unsuccessful == "error":
        raise RuntimeError("Job was not successful (not converged)!")

    raise RuntimeError(f"Unknown option for handle_unsuccessful: {handle_unsuccessful}")


def run_aims_socket(
    structures_to_calculate: list[Structure | Molecule], aims_cmd: str = None
) -> None:
    """Use the ASE interface to run FHI-aims from the socket.

    Parameters
    ----------
    structures_to_calculate: list[Structure or Molecule]
        The list of structures to run scf calculations on
    aims_cmd: str
        The aims command to use (defaults to SETTINGS.AIMS_CMD).
    """
    if aims_cmd is None:
        aims_cmd = SETTINGS.AIMS_CMD

    ase_adaptor = AseAtomsAdaptor()
    atoms_to_calculate = [
        ase_adaptor.get_atoms(structure) for structure in structures_to_calculate
    ]

    with open("parameters.json") as param_file:
        parameters = json.load(param_file, cls=MontyDecoder)

    del_keys = []
    for key, val in parameters.items():
        if val is None:
            del_keys.append(key)

    for key in del_keys:
        parameters.pop(key)

    profile = AimsProfile(command=aims_cmd if aims_cmd else SETTINGS.AIMS_CMD)

    calculator = Aims(profile=profile, **parameters)
    port = parameters["use_pimd_wrapper"][1]
    atoms = atoms_to_calculate[0].copy()

    with SocketIOCalculator(calc=calculator, port=port) as calc:
        for ac, atoms_calc in enumerate(atoms_to_calculate):
            # Delete prior calculation results
            calc.results.clear()

            # Reset atoms information to the new cell
            atoms.info = atoms_calc.info
            atoms.cell = atoms_calc.cell
            atoms.positions = atoms_calc.positions

            calc.calculate(atoms, system_changes=["positions", "cell"])
            if "plus_u_matrix_control" in parameters:
                out_mat_path = Path(f"elsi_output_calc_{ac:04d}")
                out_mat_path.mkdir(exist_ok=True)
                shutil.copy("occupation_matrix_control.txt", out_mat_path)

            if "elsi_output_matrix" in parameters:
                out_mat_path = Path(f"elsi_output_calc_{ac:04d}")
                out_mat_path.mkdir(exist_ok=True)

                for file in glob("*csc"):
                    with (
                        open(file, "rb") as f_in,
                        gzip.open(f"{out_mat_path}/{file}.gz", "wb") as f_out,
                    ):
                        shutil.copyfileobj(f_in, f_out)

        calc.close()
