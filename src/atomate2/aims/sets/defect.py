"""An input set for the FHI-aims defect (charged state) calculations."""

from __future__ import annotations

from dataclasses import dataclass

from pymatgen.io.aims.sets.core import RelaxSetGenerator


@dataclass
class ChargeStateRelaxSetGenerator(RelaxSetGenerator):
    """Generator for atomic-only relaxation for defect supercell calculations.

    Since the defect cells are assumed to be large, we will use only a single k-point.
    """

    relax_cell: bool = False
    use_structure_charge: bool = True
