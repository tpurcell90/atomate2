"""Jobs to perform the AEWF calculations in."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job

from atomate2.common.schemas.aewf import AEWFDoc

if TYPE_CHECKING:
    from emmet.core.task import BaseTaskDocument
    from pymatgen.core import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker


# Helper functions for BSON-safe float conversion
def float2bson(eta: float) -> str:
    """Convert a float to a BSON-safe entry."""
    return str(eta).replace(".", "_")


@job
def eos_check(
    jobs_outputs: dict[str, tuple[str, BaseTaskDocument]]
    | tuple[str, BaseTaskDocument, list[float]],
    relax_outputs: tuple[str, BaseTaskDocument],
    store_directory: str | Path | None = None,
    setname: str = "ae-verifcation",
    flow_uuid: str | None = None,
) -> AEWFDoc:
    """Postprocess AEWF EOS workflow.

    Parameters
    ----------
    job_outputs: dict[str, tuple[str, BaseTaskDocument]]
        The outputs for each of the EOS jobs (key: eta, value:(uuid, job output))
    relax_outputs: tuple[str, BaseTaskDocument] | None
        The outputs for the relaxation job if done (uuid, job output)
    store_directory: str | Path | None
        The optional path to store the results in outside of the job_directories
    setname: str
        Name of the dataset the workflow belongs to
    flow_uuid: str | None
        UUID for the eos workflow

    Returns
    -------
    AEWFDoc
        The TaskDocument for this workflow
    """
    task_doc = AEWFDoc.from_outputs(
        jobs_outputs, relax_outputs, setname=setname, flow_uuid=flow_uuid
    )

    if store_directory is not None:
        eos_results_path = Path(store_directory) / "eos_results"
        eos_results_path.mkdir(exist_ok=True)

        fig = task_doc.plot_eos()
        fig.savefig(f"{eos_results_path}/eos.pdf")

        with open(f"{eos_results_path}/eos.json", "w") as eos_sum:
            json.dump(task_doc.aewf_json_dict, eos_sum, indent=2)

        if task_doc.job_dirs is not None and isinstance(jobs_outputs, dict):
            for eta, job_dir in zip(
                task_doc.scaling_factors, task_doc.job_dirs.eos_jobdirs, strict=False
            ):
                shutil.copytree(
                    job_dir.split(":")[-1].strip(),
                    f"{store_directory}/scaling_{eta:.03f}/",
                    dirs_exist_ok=True,
                )
        elif task_doc.job_dirs is not None:
            shutil.copytree(
                task_doc.job_dirs.eos_jobdirs[0].split(":")[-1].strip(),
                f"{store_directory}/scaling_calcs/",
                dirs_exist_ok=True,
            )

    return task_doc


@job
def setup_eos_calculations(
    structure: Structure,
    static_maker: BaseAimsMaker,
    volume_scaling_list: list[float] | None = None,
    store_directory: str | Path | None = None,
    relax_outputs: tuple[str, BaseTaskDocument] | None = None,
    setname: str = "ae-verification",
    socket: bool = False,
) -> Response:
    """Set up all EOS calculations.

    Parameters
    ----------
    structure: Structure
        Base structure to perform the calculation on
    static_maker: BaseAimsMaker
        Maker used to calculate the energy for the volumes
    volume_scaling_lisg: list[float] | None
        The fractional values to scale the volumes by
    store_directory: str | Path | None
        The optional path to store the results in outside of the job_directories
    relax_outputs: tuple[str, BaseTaskDocument] | None
        The outputs for the relaxation job if done (uuid, job output)
    setname: str
        Name of the dataset the workflow belongs to
    socket: bool
        If True run using a socket

    Returns
    -------
    Response
        The updated workflow with all of the single-point calculations added
    """
    if volume_scaling_list is None:
        volume_scaling_list = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]

    if store_directory is not None:
        store_directory = Path(store_directory).absolute()
        store_directory.mkdir(parents=True, exist_ok=True)
        if relax_outputs is not None:
            shutil.copytree(
                relax_outputs[1].dir_name.split(":")[-1].strip(),
                f"{store_directory}/relaxation/",
                dirs_exist_ok=True,
            )

    jobs = []

    volume0 = structure.volume

    if socket:
        calc_scturctures = []
        for eta in sorted(volume_scaling_list):
            scaled_structure = structure.copy()
            scaled_structure = scaled_structure.scale_lattice(volume0 * eta)
            calc_scturctures.append(scaled_structure)
        job = static_maker.make(structure=calc_scturctures)
        job.name += " : calc_all_structs"
        jobs.append(job)
        outputs_sock = (job.uuid, job.output, sorted(volume_scaling_list))
        eos_calc_flow = Flow(jobs, output=outputs_sock)
    else:
        outputs = {}
        for eta in volume_scaling_list:
            scaled_structure = structure.copy()
            scaled_structure = scaled_structure.scale_lattice(volume0 * eta)

            job = static_maker.make(structure=scaled_structure)
            job.name += f" : scaling {eta:.02f}"
            jobs.append(job)
            outputs[str(float2bson(eta))] = (job.uuid, job.output)
        eos_calc_flow = Flow(jobs, output=outputs)

    eos_check_job = eos_check(
        eos_calc_flow.output,
        relax_outputs,
        store_directory,
        setname=setname,
        flow_uuid=eos_calc_flow.uuid,
    )

    return Response(
        replace=Flow([eos_calc_flow, eos_check_job], output=eos_check_job.output)
    )
