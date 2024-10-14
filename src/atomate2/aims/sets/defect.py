"""An input set for the FHI-aims defect (charged state) calculations."""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.io.aims.sets.core import RelaxSetGenerator, StaticSetGenerator


@dataclass
class ChargeStateRelaxSetGenerator(RelaxSetGenerator):
    """Generator for atomic-only relaxation for defect supercell calculations.
    """
    relax_cell: bool = False
    use_structure_charge: bool = True

    def __post_init__(self):
        if "species_dir" not in self.user_params:
            self.user_params["species_dir"] = "light"


@dataclass
class ChargeStateStaticSetGenerator(StaticSetGenerator):
    """Generator for static defect supercell calculations.
    """
    use_structure_charge: bool = True

    def __post_init__(self):
        if "species_dir" not in self.user_params:
            self.user_params["species_dir"] = "light"
