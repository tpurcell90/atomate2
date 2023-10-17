"""An FHI-aims jobflow runner."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from os.path import expandvars
from typing import TYPE_CHECKING

from ase.calculators.aims import Aims
from monty.json import MontyDecoder

if TYPE_CHECKING:
    from atomate2.aims.schemas.task import AimsTaskDoc
    from atomate2.aims.utils.msonable_atoms import MSONableAtoms

logger = logging.getLogger(__name__)


def run_aims(
    aims_cmd: str = None,
):
    """
    Run FHI-aims.

    Parameters
    ----------
    aims_cmd : str
        The command used to run FHI-aims (defaults to ASE_AIMS_COMMAND env variable).
    """
    if aims_cmd is None:
        aims_cmd = os.getenv("ASE_AIMS_COMMAND", "aims.x")

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


def run_aims_socket(atoms_to_calculate: list[MSONableAtoms], aims_cmd: str = None):
    """Use the ASE interface to run FHI-aims from the socket.

    Parameters
    ----------
    atoms_to_calculate: list[.MSONableAtoms]
        The list of structures to run scf calculations on
    aims_cmd: str
        The aims command to use
    """
    with open("parameters.json") as param_file:
        parameters = json.load(param_file, cls=MontyDecoder)

    if aims_cmd:
        parameters["aims_command"] = aims_cmd
    elif "aims_command" not in parameters:
        aims_cmd = os.getenv("ASE_AIMS_COMMAND", "aims.x")

    calculator = Aims(**parameters)
    port = parameters["use_pimd_wrapper"][1]
    atoms = atoms_to_calculate[0].copy()

    with calculator.socketio(port=port) as calc:
        for atoms_calc in atoms_to_calculate:
            # Delete prior calculation results
            calc.results.clear()

            # Reset atoms information to the new cell
            atoms.info = atoms_calc.info
            atoms.cell = atoms_calc.cell
            atoms.positions = atoms_calc.positions

            calc.calculate(atoms, system_changes=["positions", "cell"])

        calc.close()
