"""Defect calculation flows (adapted from the common workflow) for FHI-aims."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow.core.maker import recursive_call

from atomate2.aims.jobs.core import RelaxMaker, StaticMaker
from atomate2.aims.schemas.task import AimsTaskDoc
from atomate2.aims.sets.defect import (
    ChargeStateStaticSetGenerator,
    ChargeStateRelaxSetGenerator
)
from atomate2.common.flows import defect as defect_flows

if TYPE_CHECKING:
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    from atomate2.aims.jobs.base import BaseAimsMaker


DEFECT_KPOINT_SETTINGS = {"k_grid_density": 3}

DEFECT_RELAX_GENERATOR = ChargeStateRelaxSetGenerator(
    use_structure_charge=True,
    user_kpoints_settings=DEFECT_KPOINT_SETTINGS,
)
DEFECT_STATIC_GENERATOR = ChargeStateStaticSetGenerator(
    user_kpoints_settings=DEFECT_KPOINT_SETTINGS,
)

@dataclass
class FormationEnergyMaker(defect_flows.FormationEnergyMaker):
    """Maker class to help calculate of the formation energy diagram.

    Maker class to calculate formation energy diagrams. The main settings for
    this maker is the `relax_maker` which contains the settings for the atomic
    relaxations that each defect supercell will undergo.

    If the `validate_maker` is set to True, the maker will check for some basic
    settings in the `relax_maker` to make sure the calculations are done correctly.

    Attributes
    ----------
    defect_relax_maker: Maker
        A maker to perform an atomic-position-only relaxation on the defect charge
        states.

    bulk_relax_maker: Maker | None
        If None, the same `defect_relax_maker` will be used for the bulk supercell.
        A maker to used to perform the bulk supercell calculation.

    name: str
        The name of the flow created by this maker.

    relax_radius:
        The radius to include around the defect site for the relaxation.
        If "auto", the radius will be set to the maximum that will fit inside
        a periodic cell. If None, all atoms will be relaxed.

    perturb:
        The amount to perturb the sites in the supercell. Only perturb the
        sites with selective dynamics set to True. So this setting only works
        with `relax_radius`.

    collect_defect_entry_data: bool
        Whether to collect the defect entry data at the end of the flow.
        If True, the output of all the charge states for each symmetry distinct
        defect will be collected into a list of dictionaries that can be used
        to create a DefectEntry. The data here can be trivially combined with
        phase diagram data from the materials project API to create the formation
        energy diagrams.
    """

    defect_relax_maker: BaseAimsMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=ChargeStateRelaxSetGenerator(
                user_params={"k_grid": [1, 1, 1], "species_dir": "light"}
            )
        )
    )
    bulk_relax_maker: BaseAimsMaker | None = None
    name: str = "formation energy"

    def sc_entry_and_locpot_from_prv(
        self, previous_dir: str
    ) -> tuple[ComputedStructureEntry, dict]:
        """Copy the output structure from previous directory.

        Read the vasprun.xml file from the previous directory
        and return the structure.

        Parameters
        ----------
        previous_dir: str
            The directory to copy from.

        Returns
        -------
        ComputedStructureEntry
        """
        task_doc = AimsTaskDoc.from_directory(previous_dir)
        return task_doc.structure_entry, task_doc.calcs_reversed[0].output.locpot

    def get_planar_locpot(self, task_doc: AimsTaskDoc) -> dict:
        """Get the planar-averaged electrostatic potential."""
        return task_doc.calcs_reversed[0].output.locpot

    def validate_maker(self) -> None:
        """Check some key settings in the relax maker.

        Since this workflow is pretty complex but allows you to use any
        relax maker, it can be easy to make mistakes in the settings.
        This method should check the most important settings and raise
        an error if something is wrong.
        """

        def check_defect_relax_maker(relax_maker: RelaxMaker) -> RelaxMaker:
            input_gen = relax_maker.input_set_generator
            if not input_gen.use_structure_charge:
                raise ValueError("use_structure_charge should be set to True")
            if input_gen.relax_cell:
                raise ValueError("Cell should not be relaxed")
            return relax_maker

        recursive_call(
            self.defect_relax_maker,
            func=check_defect_relax_maker,
            class_filter=RelaxMaker,
            nested=True,
        )


@dataclass
class ConfigurationCoordinateMaker(defect_flows.ConfigurationCoordinateMaker):
    """Maker to generate a configuration coordinate diagram.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    relax_maker: BaseAimsMaker or None
        A maker to perform a atomic-position-only relaxation on the defect charge
        states.
    static_maker: BaseAimsMaker or None
        A maker to perform the single-shot static calculation of the distorted
        structures.
    distortions: tuple[float, ...]
        The distortions, as a fraction of ΔQ, to use in the calculation of the
        configuration coordinate diagram.
    """

    relax_maker: BaseAimsMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=DEFECT_RELAX_GENERATOR,
        )
    )
    static_maker: BaseAimsMaker = field(
        default_factory=lambda: StaticMaker(input_set_generator=DEFECT_STATIC_GENERATOR)
    )
    name: str = "config coordinate"