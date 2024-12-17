"""AEWF EOS workflow for FHI-aims."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from jobflow import job
from pymatgen.io.aims.sets.base import AimsInputGenerator

from atomate2.aims.jobs.core import StaticMaker
from atomate2.common.flows.aewf import BaseAEWFMaker

if TYPE_CHECKING:
    from pathlib import Path

    from pymatgen.core import Structure

    from atomate2.aims.jobs.base import BaseAimsMaker


PARAMETERS = {
    "override_warning_libxc": True,
    "xc": "pbe",
    "override_illconditioning": True,
    "basis_threshold": 1e-5,
    "relativistic": "atomic_zora scalar",
    "occupation_type": "fermi 0.061225",
    "charge": 0,
    "spin": "none",
}


class AEWFMaker(BaseAEWFMaker):
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
    """

    name: str = "AEWF EOS Workflow"
    static_maker: BaseAimsMaker = StaticMaker(
        "AEWF_EOS_STATIC", AimsInputGenerator(user_params=PARAMETERS)
    )
    relax_maker: BaseAimsMaker | None = None
    store_directory: str | Path | None = None

    @job
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
        volume0 = structure.volume
        min_scaling = scaling_factor * volume0

        lattice = structure.lattice.scale(min_scaling).reciprocal_lattice

        k_grid = [
            int(np.ceil(param / 0.12)) * 2 + 1 for param in lattice.parameters[:3]
        ]
        maker.input_set_generator.user_params["k_grid"] = k_grid

        return maker
