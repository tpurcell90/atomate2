"""Define the VASP PhononMaker."""
from dataclasses import dataclass, field
from typing import Union

from atomate2.common.flows.phonons import BasePhononMaker
from atomate2.vasp.flows.core import DoubleRelaxMaker
from atomate2.vasp.jobs.base import BaseVaspMaker
from atomate2.vasp.jobs.core import DielectricMaker, StaticMaker, TightRelaxMaker
from atomate2.vasp.jobs.phonons import PhononDisplacementMaker
from atomate2.vasp.sets.core import StaticSetGenerator


@dataclass
class PhononMaker(BasePhononMaker):
    """
    Maker to calculate harmonic phonons with VASP and Phonopy.

    Calculate the harmonic phonons of a material. Initially, a tight structural
    relaxation is performed to obtain a structure without forces on the atoms.
    Subsequently, supercells with one displaced atom are generated and accurate
    forces are computed for these structures. With the help of phonopy, these
    forces are then converted into a dynamical matrix. To correct for polarization
    effects, a correction of the dynamical matrix based on BORN charges can
    be performed.     Finally, phonon densities of states, phonon band structures
    and thermodynamic properties are computed.

    .. Note::
        It is heavily recommended to symmetrize the structure before passing it to
        this flow. Otherwise, a different space group might be detected and too
        many displacement calculations will be generated.
        It is recommended to check the convergence parameters here and
        adjust them if necessary. The default might not be strict enough
        for your specific case.

    Parameters
    ----------
    name : str
        Name of the flows produced by this maker.
    sym_reduce : bool
        Whether to reduce the number of deformations using symmetry.
    symprec : float
        Symmetry precision to use in the
        reduction of symmetry to find the primitive/conventional cell
        (use_primitive_standard_structure, use_conventional_standard_structure)
        and to handle all symmetry-related tasks in phonopy
    displacement: float
        displacement distance for phonons
    min_length: float
        min length of the supercell that will be built
    prefer_90_degrees: bool
        if set to True, supercell algorithm will first try to find a supercell
        with 3 90 degree angles
    get_supercell_size_kwargs: dict
        kwargs that will be passed to get_supercell_size to determine supercell size
    use_symmetrized_structure: str
        allowed strings: "primitive", "conventional", None

        - "primitive" will enforce to start the phonon computation
          from the primitive standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          This makes it possible to use certain k-path definitions
          with this workflow. Otherwise, we must rely on seekpath
        - "conventional" will enforce to start the phonon computation
          from the conventional standard structure
          according to Setyawan, W., & Curtarolo, S. (2010).
          High-throughput electronic band structure calculations:
          Challenges and tools. Computational Materials Science,
          49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.
          We will however use seekpath and primitive structures
          as determined by from phonopy to compute the phonon band structure
    bulk_relax_maker : .BaseVaspMaker or None
        A maker to perform a tight relaxation on the bulk.
        Set to ``None`` to skip the
        bulk relaxation
    static_energy_maker : .BaseVaspMaker or None
        A maker to perform the computation of the DFT energy on the bulk.
        Set to ``None`` to skip the
        static energy computation
    born_maker: .BaseVaspMaker or None
        Maker to compute the BORN charges.
    phonon_displacement_maker : .BaseVaspMaker or None
        Maker used to compute the forces for a supercell.
    generate_frequencies_eigenvectors_kwargs : dict
        Keyword arguments passed to :obj:`generate_frequencies_eigenvectors`.
    create_thermal_displacements: bool
        Bool that determines if thermal_displacement_matrices are computed
    kpath_scheme: str
        scheme to generate kpoints. Please be aware that
        you can only use seekpath with any kind of cell
        Otherwise, please use the standard primitive structure
        Available schemes are:
        "seekpath", "hinuma", "setyawan_curtarolo", "latimer_munro".
        "seekpath" and "hinuma" are the same definition but
        seekpath can be used with any kind of unit cell as
        it relies on phonopy to handle the relationship
        to the primitive cell and not pymatgen
    code: str
        determines the dft code. currently only vasp is implemented.
        This keyword might enable the implementation of other codes
        in the future
    store_force_constants: bool
        if True, force constants will be stored
    socket: bool
        If True, use the socket for the calculation
    """

    name: str = "phonon"
    sym_reduce: bool = True
    symprec: float = 1e-4
    displacement: float = 0.01
    min_length: Union[float, None] = 20.0
    prefer_90_degrees: bool = True
    get_supercell_size_kwargs: dict = field(default_factory=dict)
    use_symmetrized_structure: Union[str, None] = None
    create_thermal_displacements: bool = True
    generate_frequencies_eigenvectors_kwargs: dict = field(default_factory=dict)
    kpath_scheme: str = "seekpath"
    store_force_constants: bool = True
    socket: bool = False
    code: str = "vasp"
    bulk_relax_maker: Union[BaseVaspMaker, None] = field(
        default_factory=lambda: DoubleRelaxMaker.from_relax_maker(TightRelaxMaker())
    )
    static_energy_maker: Union[BaseVaspMaker, None] = field(
        default_factory=lambda: StaticMaker(
            input_set_generator=StaticSetGenerator(auto_ispin=True)
        )
    )
    born_maker: Union[BaseVaspMaker, None] = field(default_factory=DielectricMaker)
    phonon_displacement_maker: BaseVaspMaker = field(
        default_factory=PhononDisplacementMaker
    )
# TARP: Merge conflict resolution I want to see what exactly changed in the make for the new common workflow
#     def make(
#         self,
#         structure: Structure,
#         prev_vasp_dir: str | Path | None = None,
#         born: list[Matrix3D] | None = None,
#         epsilon_static: Matrix3D | None = None,
#         total_dft_energy_per_formula_unit: float | None = None,
#         supercell_matrix: Matrix3D | None = None,
#     ) -> Flow:
#         """
#         Make flow to calculate the phonon properties.

#         Parameters
#         ----------
#         structure : .Structure
#             A pymatgen structure. Please start with a structure that is nearly fully
#             optimized as the internal optimizers have very strict settings!
#         prev_vasp_dir : str or Path or None
#             A previous vasp calculation directory to use for copying outputs.
#         born: Matrix3D
#             Instead of recomputing born charges and epsilon, these values can also be
#             provided manually. If born and epsilon_static are provided, the born run
#             will be skipped it can be provided in the VASP convention with information
#             for every atom in unit cell. Please be careful when converting structures
#             within in this workflow as this could lead to errors
#         epsilon_static: Matrix3D
#             The high-frequency dielectric constant to use instead of recomputing born
#             charges and epsilon. If born, epsilon_static are provided, the born run
#             will be skipped
#         total_dft_energy_per_formula_unit: float
#             It has to be given per formula unit (as a result in corresponding Doc).
#             Instead of recomputing the energy of the bulk structure every time, this
#             value can also be provided in eV. If it is provided, the static run will be
#             skipped. This energy is the typical output dft energy of the dft workflow.
#             No conversion needed.
#         supercell_matrix: list
#             Instead of min_length, also a supercell_matrix can be given, e.g.
#             [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]
#         """
#         if self.use_symmetrized_structure not in [None, "primitive", "conventional"]:
#             raise ValueError(
#                 "use_symmetrized_structure can only be primitive, conventional, None"
#             )

#         if (
#             not self.use_symmetrized_structure == "primitive"
#             and self.kpath_scheme != "seekpath"
#         ):
#             raise ValueError(
#                 "You can only use other kpath schemes with the primitive standard "
#                 "structure"
#             )

#         if self.kpath_scheme not in [
#             "seekpath",
#             "hinuma",
#             "setyawan_curtarolo",
#             "latimer_munro",
#         ]:
#             raise ValueError("kpath scheme is not implemented")

    @property
    def prev_calc_dir_argname(self):
        """Name of argument informing static maker of previous calculation directory.

        As this differs between different DFT codes (e.g., VASP, CP2K), it
        has been left as a property to be implemented by the inheriting class.

        Note: this is only applicable if a relax_maker is specified; i.e., two
        calculations are performed for each ordering (relax -> static)
        """
        return "prev_vasp_dir"
