"""An input set for the FHI-aims defect (charged state) calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymatgen.core import Structure
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

    def get_parameter_updates(
            self,
            structure: Structure,
            prev_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Input parameter updates for calculating local ES potential"""
        abc = structure.lattice.abc
        return {"output": [f"realspace_esp {int(abc[0]/0.066)} {int(abc[1]/0.066)} {int(abc[2]/0.066)}"]}


@dataclass
class ChargeStateStaticSetGenerator(StaticSetGenerator):
    """Generator for static defect supercell calculations.
    """
    use_structure_charge: bool = True

    def __post_init__(self):
        if "species_dir" not in self.user_params:
            self.user_params["species_dir"] = "light"

    def get_parameter_updates(
            self,
            structure: Structure,
            prev_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Input parameter updates for calculating local ES potential"""
        abc = structure.lattice.abc
        return {"output": [f"realspace_esp {int(abc[0]/0.066)} {int(abc[1]/0.066)} {int(abc[2]/0.066)}"]}