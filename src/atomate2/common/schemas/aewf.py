"""Create a scehema doc for AEWF collaboration."""

import warnings
from typing import Any, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
from emmet.core.structure import StructureMetadata
from emmet.core.task import BaseTaskDocument
from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure


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

    optimization_run_uuid: Optional[str] = Field(
        None, description="optimization run uuid"
    )
    eos_workflow_uuids: Optional[list[str]] = Field(
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

    eos_jobdirs: Optional[list[Optional[str]]] = Field(
        None, description="The directories where the displacement jobs were run."
    )
    optimization_run_job_dir: Optional[str] = Field(
        None, description="Directory where optimization run was performed."
    )
    taskdoc_run_job_dir: Optional[str] = Field(
        None, description="Directory where task doc was generated."
    )


class AEWFDoc(StructureMetadata):
    """Equation of State Data for AEWF collaboration.

    Parameters
    ----------
    structure: Structure
        The structure the workflow ran on
    energies: list[float]
        List of all free energies calculated
    volumes: list[float]
        List of all volumes for the structures
    scaling_factors: list[float]
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

    structure: Structure = Field(None, description="Structure of the calculation")

    energies: list[float] = Field(
        None, description="Total free energies for all structures"
    )

    volumes: list[float] = Field(
        None, description="Total free energies for all structures"
    )

    scaling_factors: list[float] = Field(
        None, description="Scaling Factors for central volumes"
    )

    bm_fit_params: dict[str, float] = Field(
        None, description="Birch Murnaghan EOS fit parameters"
    )

    job_uuids: Optional[AEWFUUIDs] = Field(
        None, description="Job UUIDs for the workflow"
    )

    job_dirs: Optional[AEWFDirs] = Field(
        None, description="Job directories for the workflow"
    )

    @classmethod
    def from_outputs(
        cls,
        eos_outputs: dict[str, tuple[str, BaseTaskDocument]],
        relax_outputs: Optional[tuple[str, BaseTaskDocument]] = None,
    ) -> Self:
        """Get the schema from relaxation and eos outputs.

        Parameters
        ----------
        eos_outputs: dict[str, tuple[str, BaseTaskDocument]]
            uuids and outputs for the equation of state jobs (scaling_factor, output)
        relax_ouputs: Optional[tuple[str, BaseTaskDocument]]
            uuid and output for the relaxation job

        Returns
        -------
        cls
            The TaskDoc for these outputs
        """
        volumes = []
        energies = []
        scaling_factors = []
        structure = None

        relax_uuid = None
        eos_uuids = []

        relax_dir = None
        eos_jobdirs = []

        if relax_outputs:
            structure = relax_outputs[1].output.structure
            relax_dir = relax_outputs[1].dir_name
            relax_uuid = relax_outputs[0]

        for eta, eos_out in eos_outputs.items():
            task_doc = eos_out[1]
            energy = task_doc.output.free_energy
            if bson2float(eta) == 1.0:
                structure = task_doc.output.structure

            if energy is not None:
                volumes.append(task_doc.output.structure.volume)
                energies.append(energy)

                scaling_factors.append(bson2float(eta))

                eos_jobdirs.append(task_doc.dir_name)
                eos_uuids.append(eos_out[0])
            else:
                warnings.warn(
                    f"Scaling factor {eta} calculation failed (no energy).\n",
                    stacklevel=1,
                    category=Warning,
                )

        bm_fit_params = bm(volumes, energies)

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

        return cls(
            structure=structure,
            energies=energies,
            volumes=volumes,
            scaling_factors=scaling_factors,
            bm_fit_params=bm_fit_dct,
            job_uuids=job_uuids,
            job_dirs=job_dirs,
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

        uuid_mapping = {}

        if self.job_uuids is not None:
            uuid_mapping["eos_workflow"] = self.job_uuids.eos_workflow_uuids

        return {
            "bm_fit_data": bm_fit_data,
            "eos_data": eos_data,
            "uuid_mapping": uuid_mapping,
        }

    def plot_eos(
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
        fit_energies = (
            birch_murnaghan(volume_range, **self.bm_fit_params)
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
