import logging
import multiprocessing as mp
from time import perf_counter
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pathos
from floris import FlorisModel, layout_visualization as layout_viz, WindRose
from shapely.geometry import Point, Polygon


class WakeMap():
    """
    Class to calculate and plot wake maps for existing and candidate farms.
    """

    def __init__(
        self,
        fmodel: FlorisModel,
        wind_rose: WindRose,
        min_dist: float | None = None,
        boundaries: list[(float, float)] | None = None,
        candidate_cluster_layout: np.typing.NDArray | None = None,
        candidate_cluster_diameter: float | None = None,
        candidate_turbine = "iea_15MW",
        exclusion_zones: list[list[(float, float)]] = [[]],
        parallel_max_workers: int = -1,
        verbose: bool = True,
        silence_floris_warnings: bool = False,
    ):
        """
        Initialize the WakeMap object.

        Args:
            fmodel: FlorisModel object
            wind_rose: WindRose object
            min_dist: Minimum distance between turbines in meters
            boundaries: List of (x,y) tuples defining the boundaries of the area to investigate
                for candidate groups. If None, will default to a rectangle with 5 nautical mile
                clearance from the existing farm.
            candidate_cluster_layout: Layout of candidate turbines for group calculation. Should
                by a 2D numpy array with shape (n_group, 2), where each row contains the (x,y)
                location of a candidate. If None, will use a circle of diameter
                candidate_cluster_diameter to define the layout.
            candidate_cluster_diameter: Diameter of the group of turbines in meters
            candidate_turbine: Turbine type to use for candidate turbines
            exclusion_zones: List of exclusion zones, where each zone is defined by a list of
                (x,y) tuples. Exclusion zones are polygons that are not allowed to contain any
                candidate turbines. If None, will default to an empty list.
            parallel_max_workers: Maximum number of workers for parallel computation
            verbose: Verbosity flag
            silence_floris_warnings: Flag to silence FLORIS warnings
        Returns:
            Instantiated WakeMap object
        """
        self.verbose = verbose
        self.parallel_max_workers = parallel_max_workers
        self.candidate_turbine = candidate_turbine

        self.fmodel_existing = fmodel
        self.fmodel_existing.set(wind_data=wind_rose)

        if silence_floris_warnings:
            logger = logging.getLogger(name="floris")
            console_handler = logging.StreamHandler()
            logger.removeHandler(console_handler)

        self._nautical_mile = 1852 # m

        if boundaries is None:
            boundaries = [
                (self.fmodel_existing.layout_x.min() - 5*self._nautical_mile,
                 self.fmodel_existing.layout_y.min() - 5*self._nautical_mile),
                (self.fmodel_existing.layout_x.max() + 5*self._nautical_mile,
                 self.fmodel_existing.layout_y.min() + 5*self._nautical_mile),
                (self.fmodel_existing.layout_x.max() - 5*self._nautical_mile,
                 self.fmodel_existing.layout_y.max() + 5*self._nautical_mile),
                (self.fmodel_existing.layout_x.min() + 5*self._nautical_mile,
                 self.fmodel_existing.layout_y.max() - 5*self._nautical_mile)
            ]
        self.boundaries = boundaries
        self._boundary_polygon = Polygon(self.boundaries)
        self._boundary_line = self._boundary_polygon.boundary

        self.exclusion_zones = exclusion_zones
        self._exclusion_polygons = []
        for ez in self.exclusion_zones:
            self._exclusion_polygons.append(Polygon(ez))

        self.min_dist = min_dist if min_dist is not None else self._nautical_mile
        self.create_candidate_locations()

        # Create candidate cluster layout
        self.create_candidate_clusters(candidate_cluster_layout, candidate_cluster_diameter)

        self._solved = False

    @property
    def solved(self):
        return self._solved

    def certify_solved(self):
        if not self.solved:
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run"
                "compute_raw_expected_powers_serial() or "
                "compute_raw_expected_powers_parallel() first."
            )
        else:
            return None

    def create_candidate_locations(self):
        """
        Create a floris model with all candidate locations.

        Args:
            None
        Returns:
            None
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

        # Remove any points that lie within the exlusion zones
        for ez in self._exclusion_polygons:
            exclusion_mask = np.array([not ez.contains(Point(x_, y_))
                             for x_, y_ in zip(xy[:,0], xy[:,1])])
            xy = xy[exclusion_mask, :]

        # Identify xy pairs that are within limit of any existing turbine and remove
        existing_xy = np.column_stack(
            [self.fmodel_existing.layout_x, self.fmodel_existing.layout_y]
        )
        existing_mask = np.ones(xy.shape[0], dtype=bool)
        for i in range(existing_xy.shape[0]):
            existing_mask = (
                existing_mask
                & (np.linalg.norm(xy - existing_xy[i], axis=1) > self.min_dist)
            )
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

    def create_candidate_clusters(
        self,
        candidate_cluster_layout: np.typing.NDArray | None = None,
        candidate_cluster_diameter: float | None = None,
    ):
        """
        Create turbine candidate groups.

        Args:
            candidate_cluster_layout: Layout of candidate turbines for group calculation. Should
                by a 2D numpy array with shape (n_group, 2), where each row contains the (x,y)
                location of a candidate. If None, will use a circle of diameter
                candidate_cluster_diameter to define the layout.
            candidate_cluster_diameter: Diameter of the group of turbines in meters. If None and
                candidate_cluster_layout is None, will default to 3 nautical miles.
        Returns:
            None
        """
        # Check only candidate_cluster_diameter or candidate_group are provided
        if candidate_cluster_diameter is None:
            candidate_cluster_diameter = 3*self._nautical_mile

        # Disregard candidate_cluster_diameter if candidate_group supplied
        if candidate_cluster_layout is None:
            x = np.arange(0, candidate_cluster_diameter, self.min_dist)
            x, y = np.meshgrid(x, x)
            xy = np.array([x.flatten(), y.flatten()]).T
            mask = (
                np.linalg.norm(xy-xy.mean(axis=0, keepdims=True), axis=1)
                <= candidate_cluster_diameter/2
            )
            candidate_cluster_layout = xy[mask,:]

        self.candidate_layout = (
            candidate_cluster_layout
            - candidate_cluster_layout.mean(axis=0, keepdims=True)
        )
        if self.verbose:
            print("Candidate cluster created with {0} turbines.".format(
                self.candidate_layout.shape[0])
            )

    def compute_raw_expected_powers_serial(
        self,
        save_in_parts: bool = False,
        filename: str | None = None
    ):
        """
        Compute the turbine expected power for each candidate group; as well as for the existing
        farm.

        Args:
            save_in_parts: Flag to save the expected powers in parts
            filename: Filename to save to (if None and save_in_parts is True, defaults to 
                "expected_powers")
        Returns:
            None
        """
        expected_powers_existing_raw_list = []
        t_start = perf_counter()
        for i in range(self.n_candidates):
            if len(self.fmodel_existing.layout_x) > 1000:
                n_print = 1
            else:
                n_print = 10
            if self.verbose and i % n_print == 0:
                print("Computing impact on existing:", i, "of", self.n_candidates,
                      "({:.1f} s)".format(perf_counter() - t_start))
            Epower_existing = _compute_expected_powers_existing_single_external_only(
                self.fmodel_existing,
                self.fmodel_all_candidates,
                self.candidate_layout,
                i
            )
            expected_powers_existing_raw_list.append(Epower_existing)

            if save_in_parts:
                if filename is None:
                    filename = "expected_powers"
                np.savez(
                    filename + "_existing_" + str(i),
                    expected_powers_existing_raw=np.array(expected_powers_existing_raw_list),
                )
        self.expected_powers_existing_raw = np.array(expected_powers_existing_raw_list)

        if self.verbose:
            print("Computation of existing farm impacts completed in",
                  "{:.1f} s.".format(perf_counter() - t_start))
            print("Computing impact on candidates.")
        self.compute_expected_powers_candidates()

        self._solved = True

    def compute_raw_expected_powers_parallel(self):
        """
        Compute the turbine expected power for each candidate group; as well as for the existing
        farm using pathos.

        Args:
            None
        Returns:
            None
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
                (self.fmodel_existing, self.fmodel_all_candidates, self.candidate_layout, i)
            )

        t_start = perf_counter()
        if self.verbose:
            print("Computing impact on existing via parallel computation.")
        if use_pathos:
            expected_powers_existing_raw_list = pathos_pool.map(
                lambda x: _compute_expected_powers_existing_single_external_only(*x),
                parallel_inputs
            )
            pathos_pool.close()
            pathos_pool.join()
        else:
            with mp.Pool(max_workers) as p:
                expected_powers_existing_raw_list = p.starmap(
                    _compute_expected_powers_existing_single_external_only,
                    parallel_inputs
                )
        self.expected_powers_existing_raw = np.array(expected_powers_existing_raw_list)
        if self.verbose:
            print("Computation of existing farm impacts completed in",
                  "{:.1f} s.".format(perf_counter() - t_start))

        if self.verbose:
            print("Computing impact on candidates.")
        self.compute_expected_powers_candidates()

        self._solved = True

    def compute_expected_powers_candidates(self):
        """
        Compute expected power for candidates, based on FlorisModel.sample_flow_at_points().

        Args:
            None
        Returns:
            None
        """
        # Expand to all possible candidate locations.
        # TODO: Consider using ParFlorisModel (Floris v4.3)
        all_candidates_x2 = np.repeat(self.all_candidates_x, self.candidate_layout.shape[0])
        all_candidates_x2 += np.tile(self.candidate_layout[:, 0], self.all_candidates_x.shape[0])
        all_candidates_y2 = np.repeat(self.all_candidates_y, self.candidate_layout.shape[0])
        all_candidates_y2 += np.tile(self.candidate_layout[:, 1], self.all_candidates_y.shape[0])

        # Start with just a hub height wind speed (simpler than rotor averaging)
        wind_speeds = self.fmodel_existing.sample_flow_at_points(
            x=all_candidates_x2,
            y=all_candidates_y2,
            z=self.fmodel_all_candidates.reference_wind_height * np.ones_like(all_candidates_x2)
        )
        # Get power() values for those speeds
        candidate_powers = self.fmodel_all_candidates.core.farm.turbine_map[0].power_function(
            power_thrust_table=self.fmodel_all_candidates.core.farm.turbine_map[0].power_thrust_table,
            velocities=wind_speeds,
            air_density=self.fmodel_all_candidates.core.flow_field.air_density,
            yaw_angles=np.zeros_like(wind_speeds),
            tilt_angles=self.fmodel_all_candidates.core.farm.tilt_angles[0,0], # MAYBE NOT?
            tilt_interp=self.fmodel_all_candidates.core.farm.turbine_map[0].tilt_interp,
        )
        # Reshape to be size (n_findex x n_candidates x layout_size)
        candidate_powers = candidate_powers.reshape(
            self.fmodel_existing.n_findex, self.n_candidates, self.candidate_layout.shape[0]
        )
        # Apply frequencies to compute individual expected powers
        frequencies = self.fmodel_existing.wind_data.unpack_freq()
        self.expected_powers_candidates_raw = np.nansum(
            np.multiply(frequencies.reshape(-1, 1, 1), candidate_powers),
            axis=0
        )

    def process_existing_expected_powers(self, subset: list | None = None):
        """
        Compute the expected powers of existing turbines for each candidate group.

        Args:
            subset: List of indices of the existing turbines to evaluate (defaults to all existing
                turbines)
        Returns:
            expected_powers_existing: Expected power for the average turbine in the existing farm
                for each candidate group (shape n_candidates)
        """
        self.certify_solved()

        if subset is None:
            subset = list(range(self.fmodel_existing.n_turbines))

        return np.mean(self.expected_powers_existing_raw[:, subset], axis=1)


    def process_candidate_expected_powers(self):
        """
        Compute the expected powers for each candidate group.

        Args:
            None
        Returns:
            expected_powers_candidates: Expected powers for each candidate group
                (shape n_candidates)
        """
        self.certify_solved()

        return np.mean(self.expected_powers_candidates_raw, axis=1)

    def process_existing_aep_loss(self, subset: list | None = None, hours_per_year: float = 8760):
        """
        Compute the AEP loss for each candidate. Reports in GWh.

        Args:
            subset: List of indices to use for the existing turbines to evaluate
            hours_per_year: Number of hours in a year for AEP calculation
        Returns:
            existing_losses: AEP loss for the existing (subset) farm for each candidate group
                (shape n_candidates)
        """
        self.certify_solved()

        if subset is None:
            subset = list(range(self.fmodel_existing.n_turbines))

        # Run a no wake calculation for the existing turbines
        self.fmodel_existing.run_no_wake()
        aep_losses_each = (
            self.fmodel_existing.get_expected_turbine_powers()[subset].reshape(1,-1)
            - np.array(self.expected_powers_existing_raw)[:, subset]
        ) * hours_per_year / 1e9 # Report value in GWh

        existing_losses = aep_losses_each.sum(axis=1)

        return existing_losses

    def process_candidate_aep_loss(self, hours_per_year: float = 8760):
        """
        Compute the Annual Energy Production (AEP) loss for each candidate. Reports in GWh.

        Args:
            hours_per_year: Number of hours in a year for AEP calculation.
        Returns:
            candidate_group_losses: AEP loss for each candidate group (shape n_candidates)
        """
        self.certify_solved()

        # Run a no wake calculation for the candidate
        all_candidates_x2 = np.repeat(self.all_candidates_x, self.candidate_layout.shape[0])
        all_candidates_x2 += np.tile(self.candidate_layout[:, 0], self.all_candidates_x.shape[0])
        all_candidates_y2 = np.repeat(self.all_candidates_y, self.candidate_layout.shape[0])
        all_candidates_y2 += np.tile(self.candidate_layout[:, 1], self.all_candidates_y.shape[0])

        self.fmodel_all_candidates.set(
            layout_x=all_candidates_x2,
            layout_y=all_candidates_y2,
            turbine_type=[self.candidate_turbine]
        )

        self.fmodel_all_candidates.run_no_wake()
        no_wake_expected_powers = self.fmodel_all_candidates.get_expected_turbine_powers()
        no_wake_expected_powers = no_wake_expected_powers.reshape(
            self.n_candidates, self.candidate_layout.shape[0]
        )
        aep_losses_candidates = (
            no_wake_expected_powers
            - self.expected_powers_candidates_raw
        ) * hours_per_year / 1e9 # Report value in GWh

        # Revert layout
        self.fmodel_all_candidates.set(
            layout_x=self.all_candidates_x,
            layout_y=self.all_candidates_y,
        )

        candidate_group_losses = aep_losses_candidates.sum(axis=1)

        return candidate_group_losses

    def save_raw_expected_powers(self, filename: str):
        """
        Save the raw expected powers to a .npz file.
        Args:
            filename: Filename to save to
        Returns:
            None
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
        Load the raw expected powers from a .npz file.

        Args:
            filename: Filename to load
        Returns:
            None
        """
        data = np.load(filename)
        self.expected_powers_existing_raw = np.array(data["expected_powers_existing_raw"])
        self.expected_powers_candidates_raw = np.array(data["expected_powers_candidates_raw"])
        self._solved = True
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
            subset: List of indices of existing farm turbines to plot
        Returns:
            ax: Matplotlib axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        fmodel_plot = self.fmodel_existing.copy()
        if subset is not None:
            fmodel_plot.set(
                layout_x=self.fmodel_existing.layout_x[subset],
                layout_y=self.fmodel_existing.layout_y[subset],
                turbine_type=(
                    np.array(self.fmodel_existing.core.farm.turbine_definitions)[subset]
                ).tolist(),
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
        Returns:
            ax: Matplotlib axes object
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

    def plot_candidate_layout(
        self,
        candidate_idx: int,
        ax: plt.Axes | None = None,
        plotting_dict: Dict[str, Any] = {}
    ):
        """
        Plot the groups that the candidate belongs to.

        Args:
            candidate_idx: Index of the candidate group to plot
            ax: Matplotlib axes object
            plotting_dict: Dictionary of plotting options
        Returns:
            ax: Matplotlib axes object
        """
        default_plotting_dict = {
            "marker": ".",
            "markersize": 10,
            "color": "red",
            "markeredgecolor": "None"
        }
        plotting_dict = {**default_plotting_dict, **plotting_dict}
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(
            self.all_candidates_x[candidate_idx],
            self.all_candidates_y[candidate_idx],
            linestyle="None",
            **plotting_dict,
            label="Centerpoint"
        )

        ax.plot(
            self.all_candidates_x[candidate_idx] + self.candidate_layout[:, 0],
            self.all_candidates_y[candidate_idx] + self.candidate_layout[:, 1],
            label="Candidate layout",
            linestyle="None",
            alpha=0.5,
            **plotting_dict,
        )

        return ax

    def plot_exclusion_zones(
        self,
        ax: plt.Axes | None = None,
        color: str = "yellow",
        alpha: float = 0.5,
    ):
        """
        Plot the exclusion zones.

        Args:
            ax: Matplotlib axes object
            color: Color of the exclusion zones
            alpha: Alpha value for the fill
        Returns:
            ax: Matplotlib axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        for ez in self._exclusion_polygons:
            # Plot zones with border and fill
            ax.plot(*ez.exterior.xy, color=color)
            ax.fill(*ez.exterior.xy, color=color, alpha=alpha)

        return ax

    def plot_candidate_boundary(
        self,
        ax: plt.Axes | None = None,
        color: str = "black",
        alpha: float = 0.2,
    ):
        """
        Plot the boundary.

        Args:
            ax: Matplotlib axes object
            color: Color of the boundary
            alpha: Alpha value for the fill
        Returns:
            ax: Matplotlib axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(*self._boundary_line.xy, color=color)
        ax.fill(*self._boundary_polygon.exterior.xy, color=color, alpha=alpha)

        return ax

    def plot_contour(
        self,
        values: np.typing.NDArray,
        ax: plt.Axes | None = None,
        normalizer: float = 1.0,
        cmap: str | None = None,
        colorbar_label: str = ""
    ):
        """
        Create a contour plot. Mostly used as a subroutine called by higher-level
        plotting methods.

        Args:
            values: Values to plot (1D array)
            ax: Matplotlib axes object. If None, ax is created
            normalizer: Normalizer for the color scale
            cmap: Colormap to use
            colorbar_label: Label for the colorbar
        Returns:
            ax: Matplotlib axes object
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
        for x, y in zip(self.fmodel_existing.layout_x, self.fmodel_existing.layout_y):
           ax.add_artist(plt.Circle((x, y), self.min_dist, color="white"))

        ax.set_xlabel("X location [m]")
        ax.set_ylabel("Y location [m]")

        return ax

    def plot_existing_value(
        self,
        value: str = "expected_power",
        subset: list | None = None,
        ax: plt.Axes | None = None,
        normalizer: float = 1.0,
        colorbar_label: str | None = None,
        cmap: str = "Reds",
    ):
        """
        Plot the expected powers of the existing farm.

        Args:
            value: Type of value to plot. Options are "expected_power" or "aep_loss".
            subset: List of indices to use for the existing turbines to evaluate
            ax: Matplotlib axes object. If None, ax is created
            normalizer: Normalizer for the color scale
            colorbar_label: Label for the colorbar
            cmap: Colormap to use
        Returns:
            ax: Matplotlib axes object
        """
        match value: # noqa: E999
            case "expected_power" | "power":
                plot_variable = self.process_existing_expected_powers(subset=subset)
                colorbar_label_default = "Existing turbine expected power [MW]"
            case "aep_loss":
                plot_variable = self.process_existing_aep_loss(subset=subset)
                colorbar_label_default = "Existing farm AEP loss [GWh]"
            case _:
                raise ValueError(
                    "Invalid type. Must be 'power', 'expected_power', or 'aep_loss'."
                )
        colorbar_label = colorbar_label_default if colorbar_label is None else colorbar_label

        return self.plot_contour(
            plot_variable,
            ax=ax,
            normalizer=normalizer,
            cmap=cmap,
            colorbar_label=colorbar_label
        )

    def plot_candidate_value(
        self,
        value: str = "expected_power",
        ax: plt.Axes | None = None,
        normalizer: float = 1.0,
        colorbar_label: str | None = None,
        cmap: str = "Purples"
    ):
        """
        Plot the expected powers of the candidate farm.

        Args:
            value: Type of value to plot. Options are "expected_power" or "aep_loss".
            ax: Matplotlib axes object. If None, ax is created
            normalizer: Normalizer for the color scale
            colorbar_label: Label for the colorbar
            cmap: Colormap to use
        Returns:
            ax: Matplotlib axes object
        """
        match value:
            case "expected_power" | "power":
                plot_variable = self.process_candidate_expected_powers()
                colorbar_label_default = "Candidate turbine power [MW]"
            case "aep_loss":
                plot_variable = self.process_candidate_aep_loss()
                colorbar_label_default = "Candidate group AEP loss [GWh]"
            case _:
                raise ValueError(
                    "Invalid type. Must be 'power', 'expected_power', or 'aep_loss'."
                )
        colorbar_label = colorbar_label_default if colorbar_label is None else colorbar_label

        return self.plot_contour(
            plot_variable,
            ax=ax,
            normalizer=normalizer,
            cmap=cmap,
            colorbar_label=colorbar_label
        )


#### HELPER FUNCTIONS
def _compute_expected_powers_existing_single_external_only(
    fmodel_existing: FlorisModel,
    fmodel_all_candidates: FlorisModel,
    candidate_layout: np.typing.NDArray,
    candidate_idx: int
):
    """
    Compute the expected power for a single candidate group, but only considering external turbines.

    Args:
        fmodel_existing: FlorisModel object for the existing farm
        fmodel_all_candidates: FlorisModel object for all candidate locations
        candidate_layout: Layout of candidate turbines for group calculation (n_candidates x 2)
        candidate_idx: Index of the candidate group to compute
    Returns:
        expected_powers: Expected powers for the existing turbines
    """

    fmodel_candidate = fmodel_all_candidates.copy()
    fmodel_candidate.set(
        layout_x=fmodel_all_candidates.layout_x[candidate_idx] + candidate_layout[:, 0],
        layout_y=fmodel_all_candidates.layout_y[candidate_idx] + candidate_layout[:, 1],
        wind_data=fmodel_existing.wind_data
    )

    x, y, z = _sample_locations(fmodel_existing)

    wind_speeds = fmodel_candidate.sample_flow_at_points(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
    ).reshape(fmodel_existing.n_findex, *x.shape)

    # Get power() values for those speeds
    turbine_types = []
    turbine_type_names = np.array(
        [turbine.turbine_type for turbine in fmodel_existing.core.farm.turbine_map]
    )
    for turbine in fmodel_existing.core.farm.turbine_map:
        if turbine not in turbine_types:
            turbine_types.append(turbine)

    existing_powers = np.zeros((fmodel_existing.n_findex, fmodel_existing.n_turbines))
    for turbine in turbine_types:
        wind_speeds_tt = wind_speeds[:, turbine_type_names == turbine.turbine_type, : , :]
        existing_powers_tt = turbine.power_function(
            power_thrust_table=turbine.power_thrust_table,
            velocities=wind_speeds_tt,
            air_density=fmodel_existing.core.flow_field.air_density,
            yaw_angles=np.zeros((fmodel_existing.n_findex, fmodel_existing.n_turbines)),
            tilt_angles=fmodel_existing.core.farm.tilt_angles,
            tilt_interp=turbine.tilt_interp,
        )
        # assign in positions matching tt
        existing_powers[:, turbine_type_names == turbine.turbine_type] = existing_powers_tt

    # Apply frequency to compute expected powers
    frequencies = fmodel_existing.wind_data.unpack_freq()
    return np.nansum(np.multiply(frequencies.reshape(-1, 1), existing_powers), axis=0)

def _sample_locations(fmodel_existing: FlorisModel):
    """
    Provide x, y, z locations of all turbines at all rotor grid points.

    Args:
        fmodel_existing: FlorisModel object
    Returns:
        (x, y, z): Tuple of x, y, z locations of all turbines at all rotor grid points
    """

    turbine_locs = fmodel_existing.core.grid.turbine_coordinates
    radius_ratio = 0.5
    disc_area_radius = radius_ratio * fmodel_existing.core.grid.turbine_diameters / 2
    template_grid = np.ones(
        (
            fmodel_existing.core.grid.n_turbines,
            fmodel_existing.core.grid.grid_resolution,
            fmodel_existing.core.grid.grid_resolution,
        ),
        dtype=float
    )
    # Calculate the radial distance from the center of the turbine rotor.
    # If a grid resolution of 1 is selected, create a disc_grid of zeros, as
    # np.linspace would just return the starting value of -1 * disc_area_radius
    # which would place the point below the center of the rotor.
    if fmodel_existing.core.grid.grid_resolution == 1:
        disc_grid = np.zeros((np.shape(disc_area_radius)[0], 1 ))
    else:
        disc_grid = np.linspace(
            -1 * disc_area_radius,
            disc_area_radius,
            fmodel_existing.core.grid.grid_resolution,
            dtype=float,
            axis=1
        )
    # Construct the turbine grids
    # Here, they are already rotated to the correct orientation for each wind direction
    x = turbine_locs[:, 0, None, None] * template_grid

    ones_grid = np.ones(
        (fmodel_existing.core.grid.n_turbines,
         fmodel_existing.core.grid.grid_resolution,
         fmodel_existing.core.grid.grid_resolution
        ),
        dtype=float
    )
    y = turbine_locs[:, 1, None, None] + template_grid * ( disc_grid[:, :, None])
    z = turbine_locs[:, 2, None, None] + template_grid * ( disc_grid[:, None, :] * ones_grid )

    return x, y, z
