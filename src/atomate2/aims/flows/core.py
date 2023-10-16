"""(Work)flows for FHI-aims."""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

from jobflow import Flow, Maker
from pymatgen.core import Molecule, Structure

from atomate2.aims.jobs.base import BaseAimsMaker
from atomate2.aims.jobs.core import RelaxMaker
from atomate2.aims.sets.core import RelaxSetGenerator
from atomate2.aims.utils.msonable_atoms import MSONableAtoms


@dataclass
class DoubleRelaxMaker(Maker):
    """Double relaxation maker for FHI-aims.

    A maker to perform a double relaxation in FHI-aims (first with light,
    and then with tight species_defaults).

    Parameters
    ----------
    name : str
        A name for the flow
    relax_maker1: .BaseAimsMaker
        A maker that generates the first relaxation
    relax_maker2: .BaseAimsMaker
        A maker that generates the second relaxation
    """

    name: str = "Double relaxation"
    relax_maker1: BaseAimsMaker = field(default_factory=RelaxMaker)
    relax_maker2: BaseAimsMaker = field(default_factory=RelaxMaker)

    def make(
        self,
        structure: Union[MSONableAtoms, Structure, Molecule],
        prev_dir: Union[str, Path, None] = None,
    ) -> Flow:
        """Create a flow with two chained relaxations.

        Parameters
        ----------
        structure : .MSONableAtoms, Structure, Molecule
            The structure to relax.
        prev_dir : str or Path or None
            A previous FHI-aims calculation directory to copy output files from.
        """
        relax1 = self.relax_maker1.make(structure, prev_dir=prev_dir)
        relax1.name += " 1"

        relax2 = self.relax_maker2.make(
            relax1.output.atoms, prev_dir=relax1.output.dir_name
        )
        relax2.name += " 2"

        return Flow([relax1, relax2], relax2.output, name=self.name)

    @classmethod
    def from_parameters(
        cls,
        parameters: Dict[str, Any],
        species_defaults: Union[list[str], tuple[str, str]] = ("light", "tight"),
    ):
        """Create the maker from an ASE parameter set.

        Creates a DoubleRelaxFlow for the same parameters with two different
        species defaults.

        Parameters
        ----------
        parameters : dict
            a dictionary with calculation parameters
        species_defaults: list | tuple
            paths for species defaults to use relative to the given `species_dir`
            in parameters
        """
        # various checks
        if len(species_defaults) != 2:
            raise ValueError(
                "Two species defaults directories must be provided for DoubleRelaxFlow"
            )
        if "species_dir" not in parameters:
            raise KeyError("Provided parameters do not include species_dir")
        species_dir = Path(parameters["species_dir"])
        for basis_set in species_defaults:
            if not (species_dir / basis_set).exists():
                basis_set_dir = (species_dir / basis_set).as_posix()
                raise OSError(
                    f"The species defaults directory {basis_set_dir} does not exist"
                )

        # now the actual work begins
        makers = []
        for basis_set in species_defaults:
            parameters["species_dir"] = (species_dir / basis_set).as_posix()
            input_set = RelaxSetGenerator(user_parameters=deepcopy(parameters))
            makers.append(RelaxMaker(input_set_generator=input_set))
        return cls(relax_maker1=makers[0], relax_maker2=makers[1])
