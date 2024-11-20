from typing import (
    Any,
    Dict,
)

import numpy as np
import matplotlib.pyplot as plt
import pathos
import multiprocessing as mp
from time import perf_counter

from floris import FlorisModel, WindRose
from floris import layout_visualization as layout_viz

class WakeMap():
    """
    Class to calculate and plot wake maps.
    """

    def __init__(
        self,
        fmodel: FlorisModel,
        wind_rose: WindRose,
        min_dist: float | None = None,
        group_diameter: float | None = None,
        bounding_box: Dict[str, float] | None = None,
        candidate_turbine = "iea_15MW",
        parallel_max_workers: int = -1,
        verbose: bool = False
    ):
        """
        Initialize the WakeMap object.

        Args:
            fmodel: FlorisModel object
            wind_rose: WindRose object
            min_dist: Minimum distance between turbines in meters
            group_diameter: Diameter of the group of turbines in meters
            bounding_box: Dictionary of bounding box limits. Should contain keys
                "x_min", "x_max", "y_min", "y_max"
            candidate_turbine: Turbine type to use for candidate turbines
            parallel_max_workers: Maximum number of workers for parallel computation
            verbose: Verbosity flag
        """
        self.verbose = verbose
        self.parallel_max_workers = parallel_max_workers
        self.candidate_turbine = candidate_turbine

        self.fmodel_existing = fmodel
        self.fmodel_existing.set(wind_data=wind_rose)

        nautical_mile = 1852 # m

        if bounding_box is None:
            bounding_box = {
                "x_min": self.fmodel_existing.layout_x.min() - 5*nautical_mile,
                "x_max": self.fmodel_existing.layout_x.max() + 5*nautical_mile,
                "y_min": self.fmodel_existing.layout_y.min() - 5*nautical_mile,
                "y_max": self.fmodel_existing.layout_y.max() + 5*nautical_mile
            }
        self.bounding_box = bounding_box
        self.min_dist = min_dist if min_dist is not None else nautical_mile
        self.group_diameter = group_diameter if group_diameter is not None else 3*nautical_mile
        self.create_candidate_locations()

        self.create_candidate_groups()

    def create_candidate_locations(self):
        """
        Create a floris model with all candidate locations.
        """
        x_ = np.arange(
            self.bounding_box["x_min"],
            self.bounding_box["x_max"] + 0.1,
            self.min_dist
        )
        y_ = np.arange(
            self.bounding_box["y_min"],
            self.bounding_box["y_max"] + 0.1,
            self.min_dist
        )

        x, y = np.meshgrid(x_, y_)
        xy = np.column_stack([x.flatten(), y.flatten()])

        # Identify xy pairs that are within limit of any existing turbine and remove
        existing_xy = np.column_stack([self.fmodel_existing.layout_x, self.fmodel_existing.layout_y])
        mask = np.ones(xy.shape[0], dtype=bool)
        for i in range(existing_xy.shape[0]):
            mask = mask & (np.linalg.norm(xy - existing_xy[i], axis=1) > self.min_dist)
        self.all_candidates_x = xy[mask,0]
        self.all_candidates_y = xy[mask,1]

        self.fmodel_all_candidates = self.fmodel_existing.copy()
        self.fmodel_all_candidates.set(
            layout_x=self.all_candidates_x,
            layout_y=self.all_candidates_y,
            turbine_type=[self.candidate_turbine]
        )

        self.n_candidates = self.all_candidates_x.shape[0]
        if self.verbose:
            print(self.n_candidates, "candidate turbine positions created.")

    def create_candidate_groups(self):
        """
        Create turbine candidate groups.
        """
        self.groups = []
        for i in range(self.n_candidates):
            xy = np.array([self.all_candidates_x[i], self.all_candidates_y[i]])
            mask = np.linalg.norm(
                xy - np.column_stack([self.all_candidates_x, self.all_candidates_y]),
                axis=1
            ) <= self.group_diameter/2
            self.groups.append(np.where(mask)[0])

    def compute_raw_expected_powers_serial(self):
        """
        Compute the turbine expected power for each candidate group; as well as for the existing
        farm.
        """
        self.expected_powers_existing_raw = []
        t_start = perf_counter()
        for i in range(self.n_candidates):
            if self.verbose and i % 10 == 0:
                print("Computing impact on existing:", i, "of", self.n_candidates,
                      "({:.1f} s)".format(perf_counter() - t_start))
            fmodel_candidate = self.fmodel_all_candidates.copy()
            fmodel_candidate.set(
                layout_x=self.all_candidates_x[self.groups[i]],
                layout_y=self.all_candidates_y[self.groups[i]]
            )
            Epower_existing = _compute_expected_powers_existing_single(
                self.fmodel_existing,
                fmodel_candidate,
                self.fmodel_existing.wind_data
            )
            self.expected_powers_existing_raw.append(Epower_existing)

        if self.verbose:
            print("Computation of existing farm impacts completed in",
                  "{:.1f} s.".format(perf_counter() - t_start))
            print("Computing impact on candidates.")
        self._compute_expected_powers_candidates()

    def compute_raw_expected_powers_parallel(self):
        """
        Compute the turbine expected power for each candidate group; as well as for the existing
        farm using pathos.
        """
        if self.parallel_max_workers == -1:
            max_workers = pathos.helpers.cpu_count()
            if self.verbose:
                print("No maximum number of workers provided. Using CPU count ({0}).".format(
                    max_workers
                ))
        else:
            max_workers = self.parallel_max_workers
        print("here", max_workers)
        #pathos_pool = pathos.pools.ProcessPool(nodes=max_workers)
        PoolExecutor = mp.Pool
        print("here")

        # Create inputs to parallelization procedure
        parallel_inputs = []
        if self.verbose:
            print("Preparing for parallel computation.")
        for i in range(self.n_candidates):
            fmodel_candidate = self.fmodel_all_candidates.copy()
            fmodel_candidate.set(
                layout_x=self.all_candidates_x[self.groups[i]],
                layout_y=self.all_candidates_y[self.groups[i]]
            )
            parallel_inputs.append(
                (self.fmodel_existing, fmodel_candidate, self.fmodel_existing.wind_data)
            )

        t_start = perf_counter()
        if self.verbose:
            print("Computing impact on existing via parallel computation.")
        # self.expected_powers_existing_raw = pathos_pool.map(
        #     lambda x: _compute_expected_powers_existing_single(*x),
        #     parallel_inputs
        # )
        with mp.Pool(max_workers) as p:
            self.expected_powers_existing_raw = p.map(
                lambda x: _compute_expected_powers_existing_single(*x),
                parallel_inputs
            )
        # pathos_pool.close()
        # pathos_pool.join()
        if self.verbose:
            print("Computation of existing farm impacts completed in",
                  "{:.1f} s.".format(perf_counter() - t_start))

        if self.verbose:
            print("Computing impact on candidates.")
        self._compute_expected_powers_candidates()

    def _compute_expected_powers_candidates(self):
        """
        Compute expected power for candidates, based on FlorisModel.sample_flow_at_points().
        """
        # Start with just a hub height wind speed (simpler than rotor averaging)
        wind_speeds = self.fmodel_existing.sample_flow_at_points(
            x=self.all_candidates_x,
            y=self.all_candidates_y,
            z=self.fmodel_all_candidates.reference_wind_height * np.ones_like(self.all_candidates_x)
        )
        # Get power() values for those speeds
        candidate_powers = self.fmodel_all_candidates.core.farm.turbine_map[0].power_function(
            power_thrust_table=self.fmodel_all_candidates.core.farm.turbine_map[0].power_thrust_table,
            velocities=wind_speeds,
            air_density=self.fmodel_all_candidates.core.flow_field.air_density,
            yaw_angles=np.zeros_like(wind_speeds),
            tilt_angles=np.zeros_like(wind_speeds), # MAYBE NOT?
            tilt_interp=self.fmodel_all_candidates.core.farm.turbine_map[0].tilt_interp,
        )
        # Apply frequency to compute expected powers
        frequencies = self.fmodel_existing.wind_data.unpack_freq()
        self.expected_powers_candidates_raw = np.nansum(
            np.multiply(frequencies.reshape(-1, 1), candidate_powers),
            axis=0
        )

    def process_existing_expected_powers(self):
        """
        Average over all existing turbines for each candidate.
        """
        if not hasattr(self, "expected_powers_existing_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )

        return np.mean(self.expected_powers_existing_raw, axis=1)


    def process_candidate_expected_powers(self):
        """
        Is this doing anything? If so, what? Should it be? Maybe still useful?
        For now, just return the raw singular powers
        """
        if not hasattr(self, "expected_powers_candidates_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )

        # combined_powers = np.full((self.n_candidates, self.n_candidates), np.nan)
        # for i, g in enumerate(self.groups):
        #     combined_powers[i, g] = self.expected_powers_candidates_raw[i]

        # return np.nanmean(combined_powers, axis=0)
        return self.expected_powers_candidates_raw

    def process_existing_expected_powers_subset(self, subset: list):
        """
        Average over all turbines in subset for each candidate.
        """
        if not hasattr(self, "expected_powers_existing_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )

        return np.mean(np.array(self.expected_powers_existing_raw)[:, subset], axis=1)

    def save_raw_expected_powers(self, filename: str):
        """
        Save the raw expected powers to a file.
        """
        np.savez(
            filename,
            expected_powers_existing_raw=self.expected_powers_existing_raw,
            expected_powers_candidates_raw=self.expected_powers_candidates_raw
        )
        if self.verbose:
            print("Data saved.")

    def load_raw_expected_powers(self, filename: str):
        """
        Load the raw expected powers from a file.
        """
        data = np.load(filename)
        self.expected_powers_existing_raw = data["expected_powers_existing_raw"]
        self.expected_powers_candidates_raw = data["expected_powers_candidates_raw"]
        if self.verbose:
            print("Data loaded.")

    #### VISUALIZATION METHODS
    def plot_existing_farm(
        self,
        ax: plt.Axes | None = None,
        plotting_dict: Dict[str, Any] = {},
        subset: list | None = None
    ):
        """
        Plot the existing farm layout.

        Args:
            ax: Matplotlib axes object
            plotting_dict: Dictionary of plotting options
        """
        if ax is None:
            _, ax = plt.subplots()
        
        fmodel_plot = self.fmodel_existing.copy()
        if subset is not None:
            fmodel_plot.set(
                layout_x=self.fmodel_existing.layout_x[subset],
                layout_y=self.fmodel_existing.layout_y[subset]
            )

        layout_viz.plot_turbine_points(fmodel_plot, ax=ax, plotting_dict=plotting_dict)

        return ax

    def plot_candidate_locations(
        self,
        ax: plt.Axes | None = None,
        plotting_dict: Dict[str, Any] = {}
    ):
        """
        Plot the candidate locations for new turbines.

        Args:
            ax: Matplotlib axes object
            plotting_dict: Dictionary of plotting options
        """
        # Gray for candidate locations
        if "color" not in plotting_dict.keys():
            plotting_dict["color"] = "lightgray"
        
        if ax is None:
            _, ax = plt.subplots()
        layout_viz.plot_turbine_points(
            self.fmodel_all_candidates,
            ax=ax,
            plotting_dict=plotting_dict
        )

        return ax

    def plot_candidate_groups(
        self,
        candidate_idx: int,
        ax: plt.Axes | None = None,
        plotting_dict: Dict[str, Any] = {}
    ):
        """
        Plot the groups that the candidate belongs to.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.scatter(
            self.all_candidates_x[candidate_idx],
            self.all_candidates_y[candidate_idx],
            marker=".",
            s=1000,
            color="red"
        )

        plotting_dict["facecolor"] = "red"
        plotting_dict["alpha"] = 0.2
        plotting_dict["edgecolor"] = "None"
        
        for idx_c in range(self.n_candidates):
            if candidate_idx in self.groups[idx_c]:
                # Plot a filled circle centering on the idx_c location
                circle = plt.Circle(
                    (self.all_candidates_x[idx_c], self.all_candidates_y[idx_c]),
                    self.group_diameter/2,
                    **plotting_dict
                )
                ax.add_patch(circle)

        return ax

    def plot_existing_expected_powers(
        self,
        ax: plt.Axes | None = None,
        normalizer: float = 1e6,
        colorbar_label: str = "Existing turbine power [MW]"
    ):
        """
        Plot the expected powers of the existing farm.
        """
        return self.plot_power_contour(
            self.process_existing_expected_powers(),
            ax=ax,
            normalizer=normalizer,
            cmap="Blues",
            colorbar_label=colorbar_label
        )

    def plot_candidate_expected_powers(
        self,
        ax: plt.Axes | None = None,
        normalizer: float = 1e6,
        colorbar_label: str = "Candidate turbine power [MW]"
    ):
        """
        Plot the expected powers of the candidate farm.
        """
        return self.plot_power_contour(
            self.process_candidate_expected_powers(),
            ax=ax,
            normalizer=normalizer,
            cmap="Purples",
            colorbar_label=colorbar_label
        )
    
    def plot_power_contour(
        self,
        powers,
        ax: plt.Axes | None = None,
        normalizer: float = 1.0,
        cmap: str | None = None,
        colorbar_label: str = ""
    ):
        """
        Plot the expected powers of the existing farm.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        ctrf = ax.tricontourf(
            self.all_candidates_x,
            self.all_candidates_y,
            powers/normalizer,
            cmap=cmap
        )
        cbar = fig.colorbar(ctrf, ax=ax)
        cbar.set_label(colorbar_label)

        # Cover the area of the existing farm with white
        ax.tricontourf(
            self.fmodel_existing.layout_x,
            self.fmodel_existing.layout_y,
            np.ones_like(self.fmodel_existing.layout_x),
            colors="white",
        )
        ax.set_xlabel("X location [m]")
        ax.set_ylabel("Y location [m]")

        return ax

#### HELPER FUNCTIONS
def _compute_expected_powers_existing_single(fmodel_existing, fmodel_candidate, wind_rose):
    """
    Compute the expected power for a single candidate group.
    """
    fm_both = FlorisModel.merge_floris_models([fmodel_existing, fmodel_candidate])
    fm_both.set(wind_data=wind_rose)
    fm_both.run()

    return fm_both.get_expected_turbine_powers()[:fmodel_existing.layout_x.shape[0]]