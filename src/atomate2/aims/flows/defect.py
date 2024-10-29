"""Defect calculation flows (adapted from the common workflow) for FHI-aims."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow.core.maker import recursive_call

from atomate2.aims.jobs.core import RelaxMaker, StaticMaker
from atomate2.aims.schemas.calculation import AimsObject
from atomate2.aims.schemas.task import AimsTaskDoc
from atomate2.aims.sets.defect import (
    ChargeStateRelaxSetGenerator,
    ChargeStateStaticSetGenerator,
)
from atomate2.common.flows import defect as defect_flows
from atomate2.vasp.flows.defect import NonRadiativeMaker as NonRadiativeMakerVasp

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
    this maker is the `defect_relax_maker` that contain the settings for the atomic
    relaxations that each defect supercell will undergo.

    This maker can be used as a stand-alone maker to calculate all the data
    needed to populate the `DefectEntry` object. However, for you can also use this
    maker with `uc_bulk` set to True (also set `collect_defect_entry_data` to False
    and `bulk_relax_maker` to None).  This will skip the bulk supercell calculations
    assuming that bulk unit cell calculations are of high enough quality to be used
    directly.  In these cases, the bulk SC electrostatic potentials need to be
    constructed without running a separate bulk SC calculation.  This is currently
    implemented through the grid re-sampling tools in `mp-pyrho`.

    Attributes
    ----------
    defect_relax_maker: BaseAimsMaker
        A maker to perform a atomic-position-only relaxation on the defect charge
        states. Since these calculations are expensive and the settings might get
        messy, it is recommended for each implementation of this maker to check
        some of the most important settings in the `relax_maker`. Please see
        `FormationEnergyMaker.validate_maker` for more details.

    bulk_relax_maker: BaseAimsMaker
        If None, the same `defect_relax_maker` will be used for the bulk supercell.
        A maker to used to perform the bulk supercell calculation. For marginally
        converged calculations, it might be desirable to perform an additional
        lattice relaxation on the bulk supercell to make sure the energies are more
        reliable. However, if you do relax the bulk supercell, you can inadvertently
        change the grid size used in the calculation and thus the representation
        of the electrostatic potential which will affect calculation of the Freysoldt
        finite-size correction. Therefore, if you do want to perform a bulk supercell
        lattice relaxation, you should manually set the grid size.

    uc_bulk: bool
        If True, skip the bulk supercell calculation and only perform the defect
        supercell calculations. This is useful for large-scale defect databases.

    name: str
        The name of the flow created by this maker.

    relax_radius: float
        The radius to include around the defect site for the relaxation.
        If "auto", the radius will be set to the maximum that will fit inside
        a periodic cell. If None, all atoms will be relaxed.

    perturb: bool
        The amount to perturb the sites in the supercell. Only perturb the
        sites with selective dynamics set to True. So this setting only works
        with `relax_radius`.

    validate_charge: bool
        Whether to validate the charge of the defect. If True (default), the charge
        of the output structure will have to match the charge of the input defect.
        This helps catch situations where the charge of the output defect is either
        improperly set or improperly parsed before the data is stored in the
        database.

    collect_defect_entry_data: bool
        Whether to collect the defect entry data at the end of the flow.
        If True, the output of all the charge states for each symmetry distinct
        defect will be collected into a list of dictionaries that can be used
        to create a DefectEntry. The data here can be trivially combined with
        phase diagram data from the materials project API to create the formation
        energy diagrams.

        .. note::
            Once we remove the requirement for explicit bulk supercell calculations,
            this setting will be removed. It is only needed because the bulk supercell
            locpot is currently needed for the finite-size correction calculation.

        Output format for the DefectEntry data:

        .. code-block:: python

            [
                {
                    "bulk_dir_name": "computer1:/folder1",
                    "bulk_locpot": {...},
                    "bulk_uuid": "48fb6da7-dc2b-4dcb-b1c8-1203c0f72ce3",
                    "defect_dir_name": "computer1:/folder2",
                    "defect_entry": {...},
                    "defect_locpot": {...},
                    "defect_uuid": "e9af2725-d63c-49b8-a01f-391540211750",
                },
                {
                    "bulk_dir_name": "computer1:/folder3",
                    "bulk_locpot": {...},
                    "bulk_uuid": "48fb6da7-dc2b-4dcb-b1c8-1203c0f72ce3",
                    "defect_dir_name": "computer1:/folder4",
                    "defect_entry": {...},
                    "defect_locpot": {...},
                    "defect_uuid": "a1c31095-0494-4eed-9862-95311f80a993",
                },
            ]

    """

    defect_relax_maker: RelaxMaker = field(
        default_factory=lambda: RelaxMaker(
            input_set_generator=ChargeStateRelaxSetGenerator(
                user_params={"k_grid": [1, 1, 1], "species_dir": "light"}
            ),
            task_document_kwargs={"store_planar_average_data": ["realspace_ESP"]},
        )
    )
    bulk_relax_maker: RelaxMaker | None = None
    name: str = "formation energy"
    relax_radius: float | str | None = None
    perturb: float | None = None
    validate_charge: bool = True
    collect_defect_entry_data: bool = False

    def __post_init__(self) -> None:
        """Apply post init updates."""
        self.validate_maker()
        if self.uc_bulk:
            if self.bulk_relax_maker is not None:
                raise ValueError("bulk_relax_maker should be None when uc_bulk is True")
            if self.collect_defect_entry_data:
                raise ValueError(
                    "collect_defect_entry_data should be False when uc_bulk is True"
                )
        else:
            self.bulk_relax_maker = self.bulk_relax_maker or self.defect_relax_maker

    def sc_entry_and_locpot_from_prv(
        self, previous_dir: str
    ) -> tuple[ComputedStructureEntry, dict]:
        """Copy the output ComputedStructureEntry and Locpot from previous directory.

        Parameters
        ----------
        previous_dir: str
            The directory to copy from.

        Returns
        -------
        entry: ComputedStructureEntry
        """
        task_doc = AimsTaskDoc.from_directory(previous_dir)
        return task_doc.structure_entry, task_doc.aims_objects[AimsObject.LOCPOT.value]

    def get_planar_locpot(self, task_doc: AimsTaskDoc) -> dict:
        """Get the Planar Locpot from the TaskDoc.

        This is needed just in case the planar average locpot is stored in different
        part of the TaskDoc for different codes.

        Parameters
        ----------
        task_doc: AimsTaskDoc
            The task document.

        Returns
        -------
        planar_locpot: dict
            The planar average locpot.
        """
        return task_doc.aims_objects[AimsObject.LOCPOT.value]

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


class NonRadiativeMaker(NonRadiativeMakerVasp):
    """Maker to calculate non-radiative defect capture.

    Parameters
    ----------
    name: str
        The name of the flow created by this maker.
    ccd_maker: ConfigurationCoordinateMaker
        A maker to perform the calculation of the configuration coordinate diagram.
    """

    ccd_maker: ConfigurationCoordinateMaker
    name: str = "non-radiative"
