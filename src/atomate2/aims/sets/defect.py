"""An input set for the FHI-aims defect (charged state) calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pymatgen.io.aims.sets.base import AimsInputGenerator
from pymatgen.io.aims.sets.core import RelaxSetGenerator, StaticSetGenerator

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass
class ChargeStateSetGenerator(AimsInputGenerator):
    """FHI-aims input set generator for charged state calculations."""

    use_structure_charge: bool = True

    def __post_init__(self) -> None:
        """Defaults initialization."""
        if "species_dir" not in self.user_params:
            self.user_params["species_dir"] = "light"

    def get_parameter_updates(
        self, structure: Structure, prev_parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """Input parameter updates for calculating local ES potential."""
        prev_parameters = super().get_parameter_updates(structure, prev_parameters)
        abc = structure.lattice.abc
        updates = {
            "output": [
                f"realspace_esp "
                f"{int(abc[0]/0.066)} {int(abc[1]/0.066)} {int(abc[2]/0.066)}"
            ]
        }
        prev_parameters.update(updates)
        return prev_parameters


@dataclass
class ChargeStateRelaxSetGenerator(ChargeStateSetGenerator, RelaxSetGenerator):
    """Generator for atomic-only relaxation for defect supercell calculations."""

    relax_cell: bool = False


@dataclass
class ChargeStateStaticSetGenerator(ChargeStateSetGenerator, StaticSetGenerator):
    """Generator for static defect supercell calculations."""
