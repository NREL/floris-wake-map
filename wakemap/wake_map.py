from typing import (
    Any,
    Dict,
)

import numpy as np
import matplotlib.pyplot as plt

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
            min_dist_D: float | None = None,
            group_diameter_D: float | None = None,
            bounding_box: Dict[str, float] | None = None,
            parallel_compute_options: Dict[str, Any] | None = None,
            verbose: bool = False
        ):
        """
        Initialize the WakeMap object.

        Args:
            fmodel: FlorisModel object
            wind_rose: WindRose object
            min_dist_D: Minimum distance between turbines in rotor diameters
            group_diameter_D: Diameter of the group of turbines in rotor diameters
            bounding_box: Dictionary of bounding box limits. Should contain keys
                "x_min", "x_max", "y_min", "y_max"
            verbose: Verbosity flag
        """
        self.verbose = verbose
        self.parallel_compute_options = parallel_compute_options

        self.fmodel_existing = fmodel
        self.fmodel_existing.set(wind_data=wind_rose)

        # TODO: check all rotor diameters are the same
        self.D = self.fmodel_existing.core.farm.turbine_definitions[0]["rotor_diameter"]

        if bounding_box is None:
            bounding_box = {
                "x_min": self.fmodel_existing.layout_x.min() - 30*self.D,
                "x_max": self.fmodel_existing.layout_x.max() + 30*self.D,
                "y_min": self.fmodel_existing.layout_y.min() - 30*self.D,
                "y_max": self.fmodel_existing.layout_y.max() + 30*self.D
            }
        self.bounding_box = bounding_box
        self.min_dist_D = min_dist_D if min_dist_D is not None else 5
        self.group_diameter_D = group_diameter_D if group_diameter_D is not None else 22
        self.create_candidate_locations()

        self.create_candidate_groups()

    def create_candidate_locations(self):
        """
        Create a floris model with all candidate locations.

        Args:
            min_dist_D: Minimum distance between turbines in rotor diameters
            x_min_D: Minimum x-coordinate in rotor diameters
            x_max_D: Maximum x-coordinate in rotor diameters
            y_min_D: Minimum y-coordinate in rotor diameters
            y_max_D: Maximum y-coordinate in rotor diameters
        """
        x_ = np.arange(
            self.bounding_box["x_min"],
            self.bounding_box["x_max"] + 0.1,
            self.min_dist_D * self.D
        )
        y_ = np.arange(
            self.bounding_box["y_min"],
            self.bounding_box["y_max"] + 0.1,
            self.min_dist_D * self.D
        )

        x, y = np.meshgrid(x_, y_)
        xy = np.column_stack([x.flatten(), y.flatten()])

        # Identify xy pairs that are within limit of any existing turbine and remove
        existing_xy = np.column_stack([self.fmodel_existing.layout_x, self.fmodel_existing.layout_y])
        mask = np.ones(xy.shape[0], dtype=bool)
        for i in range(existing_xy.shape[0]):
            mask = mask & (np.linalg.norm(xy - existing_xy[i], axis=1) > self.min_dist_D*self.D)
        self.all_candidates_x = xy[mask,0]
        self.all_candidates_y = xy[mask,1]

        self.fmodel_all_candidates = self.fmodel_existing.copy()
        self.fmodel_all_candidates.set(
            layout_x=self.all_candidates_x,
            layout_y=self.all_candidates_y
        )

        self.n_candidates = self.all_candidates_x.shape[0]

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
            ) <= self.group_diameter_D*self.D/2
            self.groups.append(np.where(mask)[0])

    def compute_raw_expected_powers_serial(self):
        """
        Compute the turbine expected power for each candidate group; as well as for the existing
        farm.
        """
        self.expected_powers_existing_raw = []
        self.expected_powers_candidates_raw = []
        for i in range(self.n_candidates):
            if self.verbose and i % 10 == 0:
                print("Computing impact on existing:", i, "of", self.n_candidates)
            fmodel_candidate = self.fmodel_all_candidates.copy()
            fmodel_candidate.set(
                layout_x=self.all_candidates_x[self.groups[i]],
                layout_y=self.all_candidates_y[self.groups[i]]
            )
            Epower_existing, Epower_candidate = _compute_expected_powers_single(
                self.fmodel_existing,
                fmodel_candidate,
                self.fmodel_existing.wind_data
            )
            self.expected_powers_existing_raw.append(Epower_existing)
            self.expected_powers_candidates_raw.append(Epower_candidate)

    def compute_raw_expected_powers_parallel(self):
        """
        Compute the turbine expected power for each candidate group; as well as for the existing
        farm using pathos.
        """
        if self.parallel_compute_options is None:
            if self.verbose:
                print("No parallel compute options provided. Using default options.")
            # Create defaults
        pass

    def process_existing_expected_powers(self):
        """
        """
        if not hasattr(self, "expected_powers_existing_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )

        return np.mean(self.expected_powers_existing_raw, axis=1)


    def process_candidate_expected_powers(self):
        """
        """
        if not hasattr(self, "expected_powers_candidates_raw"):
            raise AttributeError(
                "FLORIS powers have not yet been computed. Please run compute_expected_powers()."
            )

        # Build big matrix for each; nanmean over them somehow. Shouldn't be too hard?
        combined_powers = np.full((self.n_candidates, self.n_candidates), np.nan)
        for i, g in enumerate(self.groups):
            combined_powers[i, g] = self.expected_powers_candidates_raw[i]

        return np.nanmean(combined_powers, axis=0)

    # Visualizations
    def plot_existing_farm(
            self,
            ax: plt.Axes | None = None,
            plotting_dict: Dict[str, Any] = {}
    ):
        """
        Plot the existing farm layout.

        Args:
            ax: Matplotlib axes object
            plotting_dict: Dictionary of plotting options
        """
        # Hardcode black for existing farm
        plotting_dict["color"] = "black"

        if ax is None:
            _, ax = plt.subplots()
        layout_viz.plot_turbine_points(self.fmodel_existing, ax=ax, plotting_dict=plotting_dict)

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
        # Hardcode gray for candidate locations
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
                    self.group_diameter_D*self.D/2,
                    **plotting_dict
                )
                ax.add_patch(circle)

        return ax

    def plot_existing_expected_powers(
            self,
            ax: plt.Axes | None = None,
            normalizer: float = 1.0
    ):
        """
        Plot the expected powers of the existing farm.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # TODO: fill locations of existing farm with NaNs for better viz.
        ctrf = ax.tricontourf(
            self.all_candidates_x,
            self.all_candidates_y,
            self.process_existing_expected_powers()/normalizer,
            cmap="Blues"
        )

        fig.colorbar(ctrf, ax=ax)

        return ax
        pass

    def plot_candidate_expected_powers(
            self,
            ax: plt.Axes | None = None,
            normalizer: float = 1.0
        ):
        """
        Plot the expected powers of the candidate farm.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        
        ctrf = ax.tricontourf(
            self.all_candidates_x,
            self.all_candidates_y,
            self.process_candidate_expected_powers()/normalizer,
            cmap="Purples"
        )

        fig.colorbar(ctrf, ax=ax)

        return ax

def _compute_expected_powers_single(fmodel_existing, fmodel_candidate, wind_rose):
    """
    Compute the expected power for a single candidate group.
    """
    fm_both = FlorisModel.merge_floris_models([fmodel_existing, fmodel_candidate])
    fm_both.set(wind_data=wind_rose)
    fm_both.run()

    all_expected_powers = fm_both.get_expected_turbine_powers()
    expected_powers_existing = all_expected_powers[:fmodel_existing.layout_x.shape[0]]
    expected_powers_candidate = all_expected_powers[fmodel_existing.layout_x.shape[0]:]

    return (expected_powers_existing, expected_powers_candidate)