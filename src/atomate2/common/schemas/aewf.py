"""Create a scehema doc for AEWF collaboration."""

import warnings
from typing import Any, Self
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from emmet.core.math import Matrix3D
from emmet.core.structure import StructureMetadata
from emmet.core.task import BaseTaskDocument
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure


import pandas as pd
import struct
import scipy.sparse as sp

import gzip

def read_elsi_to_csc(filename):
    mat = gzip.open(filename,"rb")
    data = mat.read()
    mat.close()
    i8 = "l"
    i4 = "i"

    # Get header
    start = 0
    end = 128
    header = struct.unpack(i8*16,data[start:end])

    # Number of basis functions (matrix size)
    n_basis = header[3]

    # Total number of non-zero elements
    nnz = header[5]

    # Get column pointer
    start = end
    end = start+n_basis*8
    col_ptr = struct.unpack(i8*n_basis,data[start:end])
    col_ptr += (nnz+1,)
    col_ptr = np.array(col_ptr)

    # Get row index
    start = end
    end = start+nnz*4
    row_idx = struct.unpack(i4*nnz,data[start:end])
    row_idx = np.array(row_idx)

    # Get non-zero value
    start = end

    if header[2] == 0:
        # Real case
        end = start+nnz*8
        nnz_val = struct.unpack("d"*nnz,data[start:end])
    else:
        # Complex case
        end = start+nnz*16
        nnz_val = struct.unpack("d"*nnz*2,data[start:end])
        nnz_val_real = np.array(nnz_val[0::2])
        nnz_val_imag = np.array(nnz_val[1::2])
        nnz_val = nnz_val_real + 1j*nnz_val_imag

    nnz_val = np.array(nnz_val)

    # Change convention
    for i_val in range(nnz):
        row_idx[i_val] -= 1

    for i_col in range(n_basis+1):
        col_ptr[i_col] -= 1

    return sp.csc_matrix((nnz_val,row_idx,col_ptr),shape=(n_basis,n_basis))


def bm(
    volumes: np.ndarray[float],
    energies: np.ndarray[float],
) -> tuple[float, float, float, float, float]:
    """Fit the Birch-Murnaghan equation of state.

    Parameters
    ----------
    volumes: np.ndarray[float]
        The list of all volumes to fit the data to
    energies: np.ndarray[float]
        The list of all energies to fit the data to

    Returns
    -------
    float
        The min volume
    float
        The min energy
    float
        The Bulk Modulus
    float
        The Bulk derivative
    float
        The residual error
    """
    fitdata = np.polyfit(volumes ** (-2.0 / 3.0), energies, 3, full=True)
    ssr = fitdata[1]
    sst = np.sum((energies - np.average(energies)) ** 2.0)
    residuals0 = ssr / sst
    deriv0 = np.poly1d(fitdata[0])
    deriv1 = np.polyder(deriv0, 1)
    deriv2 = np.polyder(deriv1, 1)
    deriv3 = np.polyder(deriv2, 1)

    volume0 = 0
    x = 0
    for x in np.roots(deriv1):
        # Last check: it's real, no imaginary part
        if x > 0 and deriv2(x) > 0 and abs(x.imag) < 1.0e-8:
            volume0 = x ** (-3.0 / 2.0)
            break

    if volume0 == 0:
        raise ValueError("Error: No minimum could be found")

    # Get also the min_energy and return it
    min_energy = deriv0(x)

    deriv_v2 = 4.0 / 9.0 * x**5.0 * deriv2(x)
    deriv_v3 = -20.0 / 9.0 * x ** (13.0 / 2.0) * deriv2(x) - 8.0 / 27.0 * x ** (
        15.0 / 2.0
    ) * deriv3(x)
    bulk_modulus0 = deriv_v2 / x ** (3.0 / 2.0)
    bulk_deriv0 = -1 - x ** (-3.0 / 2.0) * deriv_v3 / deriv_v2

    return volume0, min_energy, bulk_modulus0, bulk_deriv0, residuals0


def birch_murnaghan(
    volumes: np.ndarray[float],
    min_volume: float,
    min_energy: float,
    bulk_modulus: float,
    bulk_deriv: float,
) -> np.ndarray[float]:
    """Compute energy by Birch Murnaghan formula for plotting.

    Parameters
    ----------
    volumes: np.ndarray[float]
        The list of volumes to get the E fit for
    min_volume: float
        The min volume
    min_energy: float
        The min energy
    bulk_modulus: float
        The Bulk Modulus
    bulk_deriv: float
        The Bulk derivative

    Returns
    -------
    np.array[float]
        The fitted energy values
    """
    r = (min_volume / volumes) ** (2.0 / 3.0)
    return min_energy + 9.0 / 16.0 * bulk_modulus * min_volume * (r - 1.0) ** 2 * (
        2.0 + (bulk_deriv - 4.0) * (r - 1.0)
    )


def bson2float(eta: str) -> float:
    """Convert BSON float value to float.

    Parameters
    ----------
    eta: str
        The BSON value for the scaling factor

    Returns
    -------
    float
        The float of the value
    """
    return float(eta.replace("_", "."))


class AEWFUUIDs(BaseModel):
    """Collection to save all uuids connected to the AEWF run.

    Parameters
    ----------
    optimization_run_uuid: Optional[str]
        UUID for the geometry optimization calculation
    eos_workflow_uuids: Optional[list[str]]
        UUIDs for all of the EOS workflow calculaions
    """

    optimization_run_uuid: str | None = Field(None, description="optimization run uuid")
    eos_workflow_uuids: list[str] | None = Field(
        None, description="The uuids of the changed volume jobs."
    )


class AEWFDirs(BaseModel):
    """Collection to save all job directories relevant for the AEWF run.

    Parameters
    ----------
    optimization_run_jobdir: Optional[str]
        Job directory for the geometry optimization calculation
    eos_workflow_jobdirs: Optional[list[str]]
        Job directories for all of the EOS workflow calculaions
    taskdoc_run_job_dir: Optional[list[str]]
        Job directories for the EOS generation TaskDoc run
    """

    eos_jobdirs: list[str | None] | None = Field(
        None, description="The directories where the displacement jobs were run."
    )
    optimization_run_job_dir: str | None = Field(
        None, description="Directory where optimization run was performed."
    )
    taskdoc_run_job_dir: str | None = Field(
        None, description="Directory where task doc was generated."
    )


class AEWFDoc(StructureMetadata):
    """Equation of State Data for AEWF collaboration.

    Parameters
    ----------
    setname: str
        Name of the dataset the workflow belongs to
    flow_uuid: str | None
        UUID for the workflow
    structure: Structure
        The structure the workflow ran on
    energies: list[float]
        List of all free energies calculated
    volumes: list[float]
        List of all volumes for the structures
    stresses: list[Matrix3D | None]
        The stress for all structures
    x_axis_vals: list[float]
        The scaling factors for the volume and the base volumes
    bm_fit_params: dict[str, float]
        The bm fitting parameters
            {
                min_volume: The min volume
                min_energy: The min energy
                bulk_modulus: The Bulk Modulus
                bulk_deriv: The Bulk derivative
                residuals: The residual error
            }
    job_uuids: Optional[AEWFUUIDs]
        The job UUIDs
    job_dirs: Optional[AEWFDirs]
        The job running directories
    """

    setname: str = Field(
        None, description="Name of the dataset the workflow belongs to"
    )

    flow_uuid: str | None = Field(None, description="UUID for the workflow")

    structure: Structure = Field(None, description="Structure of the calculation")

    energies: list[float] = Field(
        None, description="Total free energies for all structures"
    )

    volumes: list[float] = Field(
        None, description="Total free energies for all structures"
    )

    stresses: list[Matrix3D | None] = Field(
        None, description="The stress for all structures"
    )

    x_axis_vals: list[float] = Field(
        None, description="Scaling Factors for central volumes"
    )

    bm_fit_params: dict[str, float] = Field(
        None, description="Birch Murnaghan EOS fit parameters"
    )

    job_uuids: AEWFUUIDs | None = Field(None, description="Job UUIDs for the workflow")

    job_dirs: AEWFDirs | None = Field(
        None, description="Job directories for the workflow"
    )

    density_matrix: dict[float, list[list[tuple[float, float]]]] | None = Field(None, description="The requested part of the density matrix")

    @classmethod
    def from_outputs(
        cls,
        eos_outputs: dict[str, tuple[str, BaseTaskDocument]],
        relax_outputs: tuple[str, BaseTaskDocument] | None = None,
        setname: str = "ae-verifcation",
        flow_uuid: str | None = None,
        add_den_mat: None | tuple[int, str] = None,
    ) -> Self:
        """Get the schema from relaxation and eos outputs.

        Parameters
        ----------
        eos_outputs: dict[str, tuple[str, BaseTaskDocument]]
            uuids and outputs for the equation of state jobs (scaling_factor, output)
        relax_ouputs: Optional[tuple[str, BaseTaskDocument]]
            uuid and output for the relaxation job
        setname: str
            Name of the dataset the workflow belongs to
        flow_uuid: str | None
            UUID for the workflow

        Returns
        -------
        cls
            The TaskDoc for these outputs
        """
        volumes: list[float] = []
        energies: list[float] = []
        stresses: list[Matrix3D | None] = []
        x_axis_vals: list[float] = []
        structure = None

        relax_uuid = None
        eos_uuids = []

        relax_dir = None
        eos_jobdirs = []

        if relax_outputs:
            structure = relax_outputs[1].output.structure
            relax_dir = relax_outputs[1].dir_name
            relax_uuid = relax_outputs[0]

        if isinstance(eos_outputs, dict):
            for eta, eos_out in eos_outputs.items():
                task_doc = eos_out[1]
                energy = task_doc.output.free_energy
                if bson2float(eta) == 1.0:
                    structure = task_doc.output.structure

                if energy is not None:
                    volumes.append(task_doc.output.structure.volume)
                    energies.append(energy)
                    stresses.append(task_doc.output.stress)

                    x_axis_vals.append(bson2float(eta))

                    eos_jobdirs.append(task_doc.dir_name)
                    eos_uuids.append(eos_out[0])
                else:
                    warnings.warn(
                        f"Scaling factor {eta} calculation failed (no energy).\n",
                        stacklevel=1,
                        category=Warning,
                    )
        else:
            task_doc = eos_outputs[1]
            eos_jobdirs.append(task_doc.dir_name)
            eos_uuids.append(eos_outputs[0])
            for index, eta_scale in enumerate(eos_outputs[2]):
                if eta_scale == 1.0:
                    structure = task_doc.output.trajectory[index]
                energy = task_doc.output.trajectory[index].properties.get("energy")
                print(task_doc.output.trajectory[index].properties)
                if energy is not None:
                    volumes.append(task_doc.output.trajectory[index].volume)
                    energies.append(energy)
                    stresses.append(
                        task_doc.output.trajectory[index].site_properties.get(
                            "stresses"
                        )
                    )
                    x_axis_vals.append(eta_scale)
                else:
                    warnings.warn(
                        f"Scaling factor {eta_scale} calculation failed (no energy).\n",
                        stacklevel=1,
                        category=Warning,
                    )

        bm_fit_params = bm(np.array(volumes), np.array(energies))

        bm_fit_dct = {
            "min_volume": bm_fit_params[0],
            "min_energy": bm_fit_params[1],
            "bulk_modulus": bm_fit_params[2],
            "bulk_deriv": bm_fit_params[3],
            "residuals": bm_fit_params[4],
        }

        if structure is None:
            structure = eos_out[1].output.structure.copy()
            structure = structure.scale_lattice(bm_fit_dct["min_volume"])

        job_dirs = AEWFDirs(eos_jobdirs=eos_jobdirs, optimization_run_job_dir=relax_dir)
        job_uuids = AEWFUUIDs(
            optimization_run_uuid=relax_uuid, eos_workflow_uuids=eos_uuids
        )

        density_matrix = None
        if False: # add_den_mat is not None:
            density_matrix = {}
            if not isinstance(eos_outputs, dict):
                raise ValueError("Density matrix analysis is impossible as files were overwritten")

            for xval, jobdir_host_path in zip(x_axis_vals, eos_jobdirs): 
                jobdir = jobdir_host_path.split(":")[1]
                dat = np.genfromtxt(f"{jobdir}/KS_eigenvectors.band_1.kpt_1.out")
                fxn_typ = np.genfromtxt(f"{jobdir}/KS_eigenvectors.band_1.kpt_1.out", usecols=(2,), dtype=str)
                l = np.genfromtxt(f"{jobdir}/KS_eigenvectors.band_1.kpt_1.out", usecols=(4,), dtype=str)
                
                atomic_fxns = np.where(fxn_typ == "atomic")[0]
                f_fxns = np.where(l == add_den_mat[1])[0]
                shell_fxns = np.where(dat[:, 3] == add_den_mat[0])[0]
                
                basis_fxns = np.intersect1d(np.intersect1d(atomic_fxns, f_fxns), shell_fxns)
                eig_vec_mat = dat[basis_fxns, 6::2]
                
                columns = np.arange(1, eig_vec_mat.shape[1] + 1)
                index = [f"{int(dat[ii, 0])}_{int(dat[ii, 1])}_{int(dat[ii, 5])}" for ii in basis_fxns]
                
                eig_vec_df = pd.DataFrame(data=eig_vec_mat, columns=columns, index=index)
                eig_vec_df.to_csv(f"{jobdir}/eigenvec_subshell.csv")
                
                den_mat = np.zeros((len(basis_fxns), len(basis_fxns)), dtype=np.complex128)
                xx, yy = np.meshgrid(basis_fxns, basis_fxns)
     
                eigenstate_info = np.genfromtxt(f"{jobdir}/eigenstate_info.out")
     
                for file in glob(f"{jobdir}/D_spin_01_kpt*"):
                    k_ind = int(file.split("kpt_")[1].split(".csc")[0])
                    den_mat += read_elsi_to_csc(file).toarray()[xx, yy]
     
                    den_df = pd.DataFrame(index=basis_fxns, columns=basis_fxns, data=den_mat)
                    den_df.to_csv(f"{jobdir}/density_matrix_states.csv")
                den_mat_real = np.real(den_mat)
                den_mat_imag = np.imag(den_mat)
                density_matrix[xval] = [[(den_mat_real[ii, jj], den_mat_imag[ii, jj]) for jj in range(den_mat.shape[1])] for ii in range(den_mat.shape[0])] 

        return cls(
            setname=setname,
            structure=structure,
            energies=energies,
            volumes=volumes,
            stresses=stresses,
            x_axis_vals=x_axis_vals,
            bm_fit_params=bm_fit_dct,
            job_uuids=job_uuids,
            job_dirs=job_dirs,
            flow_uuid=flow_uuid,
            density_matrix=density_matrix,
        )

    @property
    def min_volume(self) -> float:
        """The fitted minimum volume."""
        return self.bm_fit_params["min_volume"]

    @property
    def min_energy(self) -> float:
        """The fitted minimum energy."""
        return self.bm_fit_params["min_energy"]

    @property
    def bulk_modulus(self) -> float:
        """The fitted bulk modulus."""
        return self.bm_fit_params["bulk_modulus"]

    @property
    def bulk_deriv(self) -> float:
        """The fitted bulk derivative."""
        return self.bm_fit_params["bulk_deriv"]

    @property
    def residuals(self) -> float:
        """The fitted residuals."""
        return self.bm_fit_params["residuals"]

    # CODATA: 2002
    @property
    def e_charge(self) -> float:
        """The elementary charge."""
        return 1.60217653e-19

    @property
    def bulk_modulus_gpa(self) -> float:
        """The fitted bulk modulus in GPa."""
        return self.bulk_modulus * self.e_charge * 1e21

    @property
    def bulk_modulus_ev_ang3(self) -> float:
        """The fitted bulk modulus in eV / AA^3."""
        return self.bulk_modulus_gpa / 160.21766208

    @property
    def num_atoms_in_sim_cell(self) -> int:
        """Return the number of atoms in the simulation cell."""
        return self.structure.num_sites

    @property
    def aewf_json_dict(self) -> dict[str, Any]:
        """The json file for the AEWF plots."""
        bm_fit_data = {
            "min_volume": self.min_volume,
            "E0": self.min_energy,
            "bulk_modulus_ev_ang3": self.bulk_modulus_ev_ang3,
            "bulk_deriv": self.bulk_deriv,
            "residuals": self.residuals,
        }
        eos_data = [np.column_stack((self.volumes, self.energies)).tolist()]
        stress_data = [
            [vol, stress]
            for vol, stress in zip(self.volumes, self.stresses, strict=False)
        ]

        uuid_mapping: dict[str, str | list[str] | None] = {
            "eos_workflow": self.flow_uuid,
            "structure": None,
        }

        if self.job_uuids is not None:
            uuid_mapping["eos_jobs_ids"] = self.job_uuids.eos_workflow_uuids
            if self.job_uuids.optimization_run_uuid is not None:
                uuid_mapping["relax_job_id"] = self.job_uuids.optimization_run_uuid

        return {
            "bm_fit_data": bm_fit_data,
            "completely_off": [],
            "eos_data": eos_data,
            "density_matrix": self.density_matrix,
            "failed_wfs": [],
            "missing_outputs": [],
            "num_atoms_in_sim_cell": self.num_atoms_in_sim_cell,
            "script_version": "0.0.3",  # TARP Based on this
            "set_name": self.setname,
            "stress_data": stress_data,
            "uuid_mapping": uuid_mapping,
        }

    def plot(
        self, figsize: tuple[float, float] | list[float] = (3, 3), n_points: int = 301
    ) -> plt.Figure:
        """Generate a plot of the equation of state (fit and data points).

        Parameters
        ----------
        figsize: tuple[float, float] | list[float]
            The size of the figure
        n_points: int
            The number of points for the energy fig

        Returns
        -------
        plt.Figure
            the Figure of the EOS plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.tick_params(direction="in", which="both", right=True, top=True)

        volume_range = np.linspace(np.min(self.volumes), np.max(self.volumes), n_points)
        bm_fit_params = self.bm_fit_params.copy()
        bm_fit_params.pop("residuals")

        fit_energies = (
            birch_murnaghan(volume_range, **bm_fit_params)
            - self.bm_fit_params["min_energy"]
        )
        fit_energies *= 1000.0 / self.structure.num_sites

        plt_energies = (
            (np.array(self.energies) - self.bm_fit_params["min_energy"])
            * 1000.0
            / self.structure.num_sites
        )
        plt_volumes = np.array(self.volumes)

        ax.plot(plt_volumes, plt_energies, "o")
        ax.plot(volume_range, fit_energies, "k", zorder=-1)

        ax.set_xlabel("Volume [Å³]")
        ax.set_ylabel("Energy [meV / atom]")

        ax.grid = True
        fig.tight_layout()

        return fig


class AEWFParamDoc(StructureMetadata):
    """Equation of State Data for AEWF collaboration.

    Parameters
    ----------
    setname: str
        Name of the dataset the workflow belongs to
    flow_uuid: str | None
        UUID for the workflow
    structure: Structure
        The structure the workflow ran on
    energies: list[float]
        List of all free energies calculated
    input_key: str
        The key for the parameter that is changing
    x_axis_vals: list[float]
        The list of x-axis values that were tested.
    stresses: list[Matrix3D | None]
        The stress for all structures
    job_uuids: Optional[AEWFUUIDs]
        The job UUIDs
    job_dirs: Optional[AEWFDirs]
        The job running directories
    """

    setname: str = Field(
        None, description="Name of the dataset the workflow belongs to"
    )
    data_json_key: str = Field(None, description="prefix for data json keyword")
    flow_uuid: str | None = Field(None, description="UUID for the workflow")

    structure: Structure = Field(None, description="Structure of the calculation")

    energies: list[float] = Field(
        None, description="Total free energies for all structures"
    )

    fermi_energies: list[float | None] = Field(
        None, description="The Fermi energies for all structures"
    )

    input_key: str = Field(
        None, description="The key for the parameter that is changing"
    )

    x_axis_vals: list[float] = Field(
        None, description="The list of x-axis values that were tested."
    )

    job_uuids: AEWFUUIDs | None = Field(None, description="Job UUIDs for the workflow")

    job_dirs: AEWFDirs | None = Field(
        None, description="Job directories for the workflow"
    )

    @classmethod
    def from_outputs(
        cls,
        input_key: str,
        eos_outputs: dict[str, tuple[str, BaseTaskDocument]],
        relax_outputs: tuple[str, BaseTaskDocument] | None = None,
        setname: str = "ae-verifcation",
        flow_uuid: str | None = None,
    ) -> Self:
        """Get the schema from relaxation and eos outputs.

        Parameters
        ----------
        input_key: str
            The key for the parameter that changed in the inputs
        eos_outputs: dict[str, tuple[str, BaseTaskDocument]]
            uuids and outputs for the equation of state jobs (scaling_factor, output)
        relax_ouputs: Optional[tuple[str, BaseTaskDocument]]
            uuid and output for the relaxation job
        setname: str
            Name of the dataset the workflow belongs to
        flow_uuid: str | None
            UUID for the workflow

        Returns
        -------
        cls
            The TaskDoc for these outputs
        """
        x_axis_vals = []
        energies = []
        fermi_energies = []
        structure = None

        relax_uuid = None
        eos_uuids = []

        relax_dir = None
        eos_jobdirs = []
        if input_key == "fixed_spin_moment":
            data_json_key = "total_magnetization_energy"
        else:
            data_json_key = f"{input_key}_energy"

        if relax_outputs:
            structure = relax_outputs[1].output.structure
            relax_dir = relax_outputs[1].dir_name
            relax_uuid = relax_outputs[0]

        for eta, eos_out in eos_outputs.items():
            task_doc = eos_out[1]
            energy = task_doc.output.free_energy
            fermi_energy = task_doc.output.fermi_energy

            if energy is not None:
                x_axis_vals.append(task_doc.input.parameters[input_key])
                energies.append(energy)
                fermi_energies.append(fermi_energy)

                eos_jobdirs.append(task_doc.dir_name)
                eos_uuids.append(eos_out[0])
            else:
                warnings.warn(
                    f"Scaling factor {eta} calculation failed (no energy).\n",
                    stacklevel=1,
                    category=Warning,
                )

        if structure is None:
            structure = eos_out[1].output.structure.copy()

        job_dirs = AEWFDirs(eos_jobdirs=eos_jobdirs, optimization_run_job_dir=relax_dir)
        job_uuids = AEWFUUIDs(
            optimization_run_uuid=relax_uuid, eos_workflow_uuids=eos_uuids
        )

        return cls(
            data_json_key=data_json_key,
            setname=setname,
            structure=structure,
            energies=energies,
            fermi_energies=fermi_energies,
            x_axis_vals=x_axis_vals,
            job_uuids=job_uuids,
            job_dirs=job_dirs,
            flow_uuid=flow_uuid,
        )

    @property
    def num_atoms_in_sim_cell(self) -> int:
        """Return the number of atoms in the simulation cell."""
        return self.structure.num_sites

    @property
    def aewf_json_dict(self) -> dict[str, Any]:
        """The json file for the AEWF plots."""
        data = [np.column_stack((self.x_axis_vals, self.energies)).tolist()]
        fermi_energies = [
            np.column_stack((self.x_axis_vals, self.fermi_energies)).tolist()
        ]

        uuid_mapping: dict[str, str | list[str] | None] = {
            "workflow": self.flow_uuid,
            "structure": None,
        }

        if self.job_uuids is not None:
            uuid_mapping["jobs_ids"] = self.job_uuids.eos_workflow_uuids
            if self.job_uuids.optimization_run_uuid is not None:
                uuid_mapping["relax_job_id"] = self.job_uuids.optimization_run_uuid

        return {
            "script_version": "0.0.3",  # TARP Based on this
            "set_name": self.setname,
            f"{self.data_json_key}_data": data,
            "fermi_energies": fermi_energies,
            "uuid_mapping": uuid_mapping,
        }

    def plot(
        self, figsize: tuple[float, float] | list[float] = (3, 3), n_points: int = 301
    ) -> plt.Figure:
        """Generate a plot of the equation of state (fit and data points).

        Parameters
        ----------
        figsize: tuple[float, float] | list[float]
            The size of the figure
        n_points: int
            The number of points for the energy fig

        Returns
        -------
        plt.Figure
            the Figure of the EOS plot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
        ax.tick_params(direction="in", which="both", right=True, top=True)

        poly_coeff = np.polyfit(self.x_axis_vals, self.energies, 2)
        x_range = np.linspace(
            np.min(self.x_axis_vals), np.max(self.x_axis_vals), n_points
        )

        fit_energies = (
            x_range * poly_coeff[1] + x_range**2.0 * poly_coeff[0] + poly_coeff[2]
        )
        fit_energies *= 1000.0 / self.structure.num_sites

        plt_energies = (np.array(self.energies)) * 1000.0 / self.structure.num_sites
        plt_x_vals = np.array(self.x_axis_vals)

        ax.plot(plt_x_vals, plt_energies, "o")
        ax.plot(x_range, fit_energies, "k", zorder=-1)

        ax.set_xlabel(self.data_json_key)
        ax.set_ylabel("Energy [meV / atom]")

        ax.grid = True
        fig.tight_layout()

        return fig
