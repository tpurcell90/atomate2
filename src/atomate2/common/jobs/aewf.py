"""Jobs to perform the AEWF calculations in."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from jobflow import Flow, Response, job

from atomate2.common.schemas.aewf import AEWFDoc, AEWFParamDoc

if TYPE_CHECKING:
    from emmet.core.task import BaseTaskDocument
    from pymatgen.core import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker


# Helper functions for BSON-safe float conversion
def float2bson(val: float) -> str:
    """Convert a float to a BSON-safe entry."""
    return str(val).replace(".", "_")


@job
def eos_check(
    input_key: str,
    jobs_outputs: dict[str, tuple[str, BaseTaskDocument]],
    relax_outputs: tuple[str, BaseTaskDocument],
    store_directory: str | Path | None = None,
    setname: str = "ae-verifcation",
    flow_uuid: str | None = None,
) -> AEWFDoc | AEWFParamDoc:
    """Postprocess AEWF EOS workflow.

    Parameters
    ----------
    input_key: str
        The input_key for the parameter that is changing
    job_outputs: dict[str, tuple[str, BaseTaskDocument]]
        The outputs for each of the EOS jobs (key: val, value:(uuid, job output))
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
    AEWFDoc | AEWFParamDoc
        The TaskDocument for this workflow
    """
    if input_key == "volume_scaling":
        task_doc = AEWFDoc.from_outputs(
            jobs_outputs, relax_outputs, setname=setname, flow_uuid=flow_uuid
        )
    else:
        task_doc = AEWFParamDoc.from_outputs(
            input_key, jobs_outputs, relax_outputs, setname=setname, flow_uuid=flow_uuid
        )

    if store_directory is not None:
        results_path = Path(store_directory) / "results"
        results_path.mkdir(exist_ok=True)

        fig = task_doc.plot()
        fig.savefig(f"{results_path}/verification.pdf")

        with open(f"{results_path}/data.json", "w") as eos_sum:
            json.dump(task_doc.aewf_json_dict, eos_sum, indent=2)

        if task_doc.job_dirs is not None and isinstance(jobs_outputs, dict):
            for val, job_dir in zip(
                task_doc.x_axis_vals, task_doc.job_dirs.eos_jobdirs, strict=False
            ):
                shutil.copytree(
                    job_dir.split(":")[-1].strip(),
                    f"{store_directory}/x_val_{val:.03f}/",
                    dirs_exist_ok=True,
                )
        elif task_doc.job_dirs is not None:
            shutil.copytree(
                task_doc.job_dirs.eos_jobdirs[0].split(":")[-1].strip(),
                f"{store_directory}/x_val_calcs/",
                dirs_exist_ok=True,
            )

    return task_doc


@job
def setup_eos_calculations(
    structure: Structure,
    static_maker: BaseAimsMaker,
    input_key: str = "volume_scaling",
    x_axis_values: list[float] | None = None,
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
    input_key: str
        The key used to describe the change in property for the x-axis
    x_axis_values: list[float] | None
        The fractional values to set the input key too
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
    if x_axis_values is None and input_key == "volume_scaling":
        x_axis_values = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
    elif x_axis_values is None:
        raise ValueError(
            f"No default for x_axis_values is known from input parameter {input_key}."
        )

    if socket and input_key != "volume_scaling":
        raise ValueError("Socket calculations are only available for volume_scaling")

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
        for val in sorted(x_axis_values):
            scaled_structure = structure.copy()
            scaled_structure = scaled_structure.scale_lattice(volume0 * val)
            calc_scturctures.append(scaled_structure)
        job = static_maker.make(structure=calc_scturctures)
        job.name += " : calc_all_structs"
        jobs.append(job)
        outputs_sock = (job.uuid, job.output, sorted(x_axis_values))
        eos_calc_flow = Flow(jobs, output=outputs_sock)
    else:
        outputs = {}
        for val in x_axis_values:
            job_struct = structure.copy()
            if input_key == "volume_scaling":
                job_struct = job_struct.scale_lattice(volume0 * val)
            else:
                static_maker.input_set_generator.user_params[input_key] = val
            job = static_maker.make(structure=job_struct)
            job.name += f" : scaling {val:.02f}"
            jobs.append(job)
            outputs[str(float2bson(val))] = (job.uuid, job.output)
        eos_calc_flow = Flow(jobs, output=outputs)

    eos_check_job = eos_check(
        input_key,
        eos_calc_flow.output,
        relax_outputs,
        store_directory=store_directory,
        setname=setname,
        flow_uuid=eos_calc_flow.uuid,
    )
    return Response(
        replace=Flow([eos_calc_flow, eos_check_job], output=eos_check_job.output)
    )
