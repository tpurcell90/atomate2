"""Maker for the AEWF collaboration."""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jobflow import Flow, Maker, job
from pymatgen.core import Structure

from atomate2.aims.jobs.base import BaseAimsMaker
from atomate2.common.jobs.aewf import setup_eos_calculations


@dataclass
class BaseAEWFMaker(Maker):
    """Maker for AEWF EOS Calculations.

    Parameters
    ----------
    name: str
        Name for the workflow maker
    static_maker: BaseAimsMaker
        EOS volume maker
    relax_maker: BaseAimsMaker | None
        Relaxation maker for finding a good central volume
    store_directory: str | Path |None
        Directory to store the results in
    setname: str
        Name of the dataset the workflow belongs to
    """

    name: str = "AEWF EOS Workflow"
    static_maker: BaseAimsMaker = None
    relax_maker: BaseAimsMaker | None = None
    store_directory: str | Path | None = None
    scaling_factors: list[float] | None = None
    setname: str = None

    def __post_init__(self) -> None:
        """Set the default scaling factors."""
        if self.scaling_factors is None:
            self.scaling_factors = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]

    def make(
        self,
        structure: Structure,
        prev_dir: str | Path | None = None,
        subdirec: str | Path | None = None,
        socket: bool = False,
    ) -> Flow:
        """Create an AEWF EOS calculation.

        Parameters
        ----------
        structure: Structure
            The structure to run the calculation for
        prev_dir: str | Path | None
            The previous calculation directory
        subdirec: str | Path | None
            The subdirectory to store the results of the calculation
        socket: bool
            If True use socket calculators

        Returns
        -------
        Flow
            The workflow to be calculated
        """
        structure_eos = structure.copy()
        jobs = []
        relax_outputs = None
        if self.relax_maker is not None:
            relax_job = self.relax_maker.make(structure)
            jobs.append(relax_job)

            structure_eos = relax_job.output.structure
            relax_outputs = (relax_job.uuid, relax_job.output)

        jobs.append(
            self.update_kgrid(
                structure_eos, self.static_maker, np.min(self.scaling_factors)
            )
        )
        static_maker = jobs[-1].output
        store_directory = Path(self.store_directory)
        if subdirec is not None:
            store_directory = store_directory / subdirec

        eos_static_calcs = setup_eos_calculations(
            structure_eos,
            static_maker,
            volume_scaling_list=self.scaling_factors,
            store_directory=store_directory,
            relax_outputs=relax_outputs,
            setname=self.setname,
            socket=socket,
        )

        jobs.append(eos_static_calcs)

        return Flow(jobs, output=eos_static_calcs.output)

    @job
    @abstractmethod
    def update_kgrid(
        self, structure: Structure, maker: BaseAimsMaker, scaling_factor: float = 0.94
    ) -> BaseAimsMaker:
        """Update the k_grid in a maker.

        Parameters
        ----------
        structure: Structure
            The structure for the calculation
        maker: BaseAimsMaker
            The Maker to update
        scaling_factor:float
            The scaling factor for the lattice

        Returns
        -------
        BaseAimsMaker
            The updated Maker
        """
        return maker
