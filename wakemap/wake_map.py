from typing import (
    Any,
    Dict,
)

import numpy as np
import matplotlib.pyplot as plt
import pathos
import multiprocessing as mp
from time import perf_counter
from shapely.geometry import Polygon, Point

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
        boundaries: list[(float, float)] | None = None,
        candidate_turbine = "iea_15MW",
        exclusion_zones: list[list[(float, float)]] = [[]],
        parallel_max_workers: int = -1,
        external_losses_only: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the WakeMap object.

        Args:
            fmodel: FlorisModel object
            wind_rose: WindRose object (is this needed? Could come on fmodel?)
            min_dist: Minimum distance between turbines in meters
            group_diameter: Diameter of the group of turbines in meters
            bounding_box: Dictionary of bounding box limits. Should contain keys
                "x_min", "x_max", "y_min", "y_max"
            candidate_turbine: Turbine type to use for candidate turbines
            parallel_max_workers: Maximum number of workers for parallel computation
            external_losses_only: Flag to compute only the external losses for existing turbines.
                This speeds up computation.
            verbose: Verbosity flag
        """
        self.verbose = verbose
        self.parallel_max_workers = parallel_max_workers
        self.candidate_turbine = candidate_turbine

        self.fmodel_existing = fmodel
        self.fmodel_existing.set(wind_data=wind_rose)

        nautical_mile = 1852 # m

        if boundaries is None:
            boundaries = [
                (self.fmodel_existing.layout_x.min() - 5*nautical_mile,
                 self.fmodel_existing.layout_y.min() - 5*nautical_mile),
                (self.fmodel_existing.layout_x.max() + 5*nautical_mile,
                 self.fmodel_existing.layout_y.min() + 5*nautical_mile),
                (self.fmodel_existing.layout_x.max() - 5*nautical_mile,
                 self.fmodel_existing.layout_y.max() + 5*nautical_mile),
                (self.fmodel_existing.layout_x.min() + 5*nautical_mile,
                 self.fmodel_existing.layout_y.max() - 5*nautical_mile)
            ]
        self.boundaries = boundaries
        self._boundary_polygon = Polygon(self.boundaries)
        self._boundary_line = self._boundary_polygon.boundary


        self.min_dist = min_dist if min_dist is not None else nautical_mile
        self.group_diameter = group_diameter if group_diameter is not None else 3*nautical_mile
        self.create_candidate_locations()

        self.create_candidate_groups()

        self._compute_existing_single_function = (
            _compute_expected_powers_existing_single_external_only if external_losses_only
            else _compute_expected_powers_existing_single
        )

    def create_candidate_locations(self):
        """
        Create a floris model with all candidate locations.
        """
        x, y = self._boundary_polygon.exterior.coords.xy

        x_ = np.arange(
            min(x),
            max(x) + 0.1,
            self.min_dist
        )
        y_ = np.arange(
            min(y),
            max(y) + 0.1,
            self.min_dist
        )

        x, y = np.meshgrid(x_, y_)
        
        # Find all x, y pairs that lie within the boundary polygon
        boundary_mask = np.array([self._boundary_polygon.contains(Point(x_, y_))
                                  for x_, y_ in zip(x.flatten(), y.flatten())])

        xy = np.column_stack([x.flatten()[boundary_mask], y.flatten()[boundary_mask]])

        # Identify xy pairs that are within limit of any existing turbine and remove
        existing_xy = np.column_stack([self.fmodel_existing.layout_x, self.fmodel_existing.layout_y])
        existing_mask = np.ones(xy.shape[0], dtype=bool)
        for i in range(existing_xy.shape[0]):
            existing_mask = existing_mask & (np.linalg.norm(xy - existing_xy[i], axis=1) > self.min_dist)
        self.all_candidates_x = xy[existing_mask,0]
        self.all_candidates_y = xy[existing_mask,1]

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

    def compute_raw_expected_powers_serial(self, save_in_parts=False, filename=None):
        """
        Compute the turbine expected power for each candidate group; as well as for the existing
        farm.
        """
        self.expected_powers_existing_raw = []
        t_start = perf_counter()
        for i in range(self.n_candidates):
            if len(self.fmodel_existing.layout_x) > 1000:
                n_print = 1
            else:
                n_print = 10
            if self.verbose and i % n_print == 0:
                print("Computing impact on existing:", i, "of", self.n_candidates,
                      "({:.1f} s)".format(perf_counter() - t_start))
            Epower_existing = self._compute_existing_single_function(
                self.fmodel_existing,
                self.fmodel_all_candidates,
                self.groups[i]
            )
            self.expected_powers_existing_raw.append(Epower_existing)

            if save_in_parts:
                np.savez(
                    filename + "_existing_" + str(i),
                    expected_powers_existing_raw=np.array(self.expected_powers_existing_raw),
                )

        if self.verbose:
            print("Computation of existing farm impacts completed in",
                  "{:.1f} s.".format(perf_counter() - t_start))
            print("Computing impact on candidates.")
        self.compute_expected_powers_candidates()

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

        use_pathos = True # Hardcode for now. False will use multiprocessing
        
        if use_pathos:
            pathos_pool = pathos.pools.ProcessPool(nodes=max_workers)

        # Create inputs to parallelization procedure
        parallel_inputs = []
        if self.verbose:
            print("Preparing for parallel computation.")
        for i in range(self.n_candidates):
            parallel_inputs.append(
                (self.fmodel_existing, self.fmodel_all_candidates, self.groups[i])
            )

        t_start = perf_counter()
        if self.verbose:
            print("Computing impact on existing via parallel computation.")
        if use_pathos:
            self.expected_powers_existing_raw = pathos_pool.map(
                lambda x: self._compute_existing_single_function(*x),
                parallel_inputs
            )
            pathos_pool.close()
            pathos_pool.join()
        else:
            with mp.Pool(max_workers) as p:
                self.expected_powers_existing_raw = p.starmap(
                    self._compute_existing_single_function,
                    parallel_inputs
                )
        if self.verbose:
            print("Computation of existing farm impacts completed in",
                  "{:.1f} s.".format(perf_counter() - t_start))

        if self.verbose:
            print("Computing impact on candidates.")
        self.compute_expected_powers_candidates()

    def compute_expected_powers_candidates(self):
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
    
    def process_existing_expected_capacity_factors(self):
        """
        Average over all existing turbines for each candidate.
        """
        if not hasattr(self, "expected_powers_existing_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )

        rated_powers = np.array(
            [turbine.power_thrust_table["power"].max()
             for turbine in self.fmodel_existing.core.farm.turbine_map]
        ).reshape(1, -1)*1e3

        return np.mean(np.array(self.expected_powers_existing_raw)/rated_powers, axis=1)
    
    def process_candidate_expected_capacity_factors(self):
        """
        """
        if not hasattr(self, "expected_powers_candidates_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )

        # Only type of candidate turbine
        rated_power = self.fmodel_all_candidates.core.farm.turbine_map[0]\
            .power_thrust_table["power"].max()*1e3

        return self.expected_powers_candidates_raw/rated_power

    def process_existing_expected_capacity_factors_subset(self, subset: list):
        """
        Average over all turbines in subset for each candidate.
        """
        if not hasattr(self, "expected_powers_existing_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )
        rated_powers = np.array(
            [turbine.power_thrust_table["power"].max()
             for turbine in np.array(self.fmodel_all_candidates.core.farm.turbine_map)[subset]]
        ).reshape(1, -1)*1e3

        return np.mean(np.array(self.expected_powers_existing_raw)[:, subset]/rated_powers, axis=1)

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

    def plot_existing_value(
        self,
        value: str = "power",
        ax: plt.Axes | None = None,
        normalizer: float = 1.0,
        colorbar_label: str = "Existing turbine power [MW]"
    ):
        """
        Plot the expected powers of the existing farm.
        """
        if value == "power":
            plot_variable = self.process_existing_expected_powers()
        elif value == "capacity_factor":
            plot_variable = self.process_existing_expected_capacity_factors()
        else:
            raise ValueError("Invalid type. Must be 'power' or 'capacity_factor'.")
        
        return self.plot_contour(
            plot_variable,
            ax=ax,
            normalizer=normalizer,
            cmap="Blues",
            colorbar_label=colorbar_label
        )

    def plot_candidate_value(
        self,
        value: str = "power",
        ax: plt.Axes | None = None,
        normalizer: float = 1.0,
        colorbar_label: str | None = None,
    ):
        """
        Plot the expected powers of the candidate farm.
        """
        if value == "power":
            plot_variable = self.process_candidate_expected_powers()
            if colorbar_label is None:
                colorbar_label = "Candidate turbine power [MW]"
        elif value == "capacity_factor":
            plot_variable = self.process_candidate_expected_capacity_factors()
            if colorbar_label is None:
                colorbar_label = "Candidate turbine capacity factor [-]"
        else:
            raise ValueError("Invalid type. Must be 'power' or 'capacity_factor'.")
        
        return self.plot_contour(
            plot_variable,
            ax=ax,
            normalizer=normalizer,
            cmap="Purples",
            colorbar_label=colorbar_label
        )
    
    def plot_contour(
        self,
        values,
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
            values/normalizer,
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
def _compute_expected_powers_existing_single(fmodel_existing, fmodel_candidates_all, group):
    """
    Compute the expected power for a single candidate group.
    """
    fmodel_candidate = fmodel_candidates_all.copy()
    fmodel_candidate.set(
        layout_x=fmodel_candidates_all.layout_x[group],
        layout_y=fmodel_candidates_all.layout_y[group]
    )
    fmodel_both = FlorisModel.merge_floris_models([fmodel_existing, fmodel_candidate])
    fmodel_both.set(wind_data=fmodel_existing.wind_data)
    fmodel_both.run()

    return fmodel_both.get_expected_turbine_powers()[:fmodel_existing.layout_x.shape[0]]

def _compute_expected_powers_existing_single_external_only(
    fmodel_existing,
    fmodel_candidates_all,
    group
):
    """
    Compute the expected power for a single candidate group, but only considering external turbines.
    """

    fmodel_candidate = fmodel_candidates_all.copy()
    fmodel_candidate.set(
        layout_x=fmodel_candidates_all.layout_x[group],
        layout_y=fmodel_candidates_all.layout_y[group],
        wind_data=fmodel_existing.wind_data
    )

    wind_speeds = fmodel_candidate.sample_flow_at_points(
        x=fmodel_existing.layout_x,
        y=fmodel_existing.layout_y,
        z=fmodel_existing.reference_wind_height * np.ones_like(fmodel_existing.layout_x)
    )

    # Get power() values for those speeds
    turbine_types = []
    turbine_type_names = np.array(
        [turbine.turbine_type for turbine in fmodel_existing.core.farm.turbine_map]
    )
    for turbine in fmodel_existing.core.farm.turbine_map:
        if turbine not in turbine_types:
            turbine_types.append(turbine)

    existing_powers = np.zeros_like(wind_speeds)
    for turbine in turbine_types:
        wind_speeds_tt = wind_speeds[:, turbine_type_names == turbine.turbine_type]
        existing_powers_tt = turbine.power_function(
            power_thrust_table=turbine.power_thrust_table,
            velocities=wind_speeds_tt,
            air_density=fmodel_existing.core.flow_field.air_density,
            yaw_angles=np.zeros_like(wind_speeds_tt),
            tilt_angles=np.zeros_like(wind_speeds_tt), # MAYBE NOT?
            tilt_interp=turbine.tilt_interp,
        )
        # assign in positions matching tt
        existing_powers[:, turbine_type_names == turbine.turbine_type] = existing_powers_tt

    existing_powers = fmodel_existing.core.farm.turbine_map[0].power_function(
        power_thrust_table=fmodel_existing.core.farm.turbine_map[0].power_thrust_table,
        velocities=wind_speeds,
        air_density=fmodel_existing.core.flow_field.air_density,
        yaw_angles=np.zeros_like(wind_speeds),
        tilt_angles=np.zeros_like(wind_speeds), # MAYBE NOT?
        tilt_interp=fmodel_existing.core.farm.turbine_map[0].tilt_interp,
    )

    # Apply frequency to compute expected powers
    frequencies = fmodel_existing.wind_data.unpack_freq()
    return np.nansum(np.multiply(frequencies.reshape(-1, 1), existing_powers), axis=0)