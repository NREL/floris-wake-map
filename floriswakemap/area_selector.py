from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from floris import layout_visualization as layout_viz

from floriswakemap import WakeMap


class AreaSelector():
    """
    Class for identifying candidate regions for new development based on a WakeMap.
    """

    def __init__(self, wake_map: WakeMap, verbose: bool = True):
        """
        Constructor for the AreaSelector class.

        Receives instantiated WakeMap object and checks that the main maps have been
        computed.

        Args:
            wake_map: The WakeMap object to use for area selection.
            verbose: verbosity flag
        Returns:
            Instantiated AreaSelector object.
        """

        if not wake_map.solved:
            raise RuntimeError("WakeMap must be solved before creating an AreaSelector.")

        self.wake_map = wake_map

        self._allowable_candidates_x = np.copy(self.wake_map.all_candidates_x)
        self._allowable_candidates_y = np.copy(self.wake_map.all_candidates_y)

        self.verbose = verbose

        self.reset_constraints()
        self.reset_objective()

        self._state = 0 # Initialized

    def add_constraint(self, constraint_dict: Dict[str, Any]):
        """
        Add value constraint to the area selection process.

        Args:
            constraint_dict: Dictionary containing the constraint information.
                Must contain keys: turbines, value, threshold, name
                turbines: "existing" or "candidates"
                value: "expected_power" or "aep_loss"
                threshold: float value for the constraint (upper limit for "expected_power", upper
                    limit for "aep_loss")
                name: string name for the constraint
                subset: list of indices for the subset of existing turbines to apply the constraint
                    to (optional)
        Returns:
            None
        """

        required_keys = ["turbines", "value", "threshold", "name"]

        # Check that all inputs are valid
        if not all([key in constraint_dict for key in required_keys]):
            raise ValueError("Constraint dictionary must contain keys: {0}".format(required_keys))

        if constraint_dict["turbines"] not in ["existing", "candidates"]:
            raise ValueError(
                "Constraint dictionary key 'turbines' must be either 'existing' or 'candidate'."
            )

        if "subset" in constraint_dict.keys():
            if not isinstance(constraint_dict["subset"], list):
                raise ValueError("Constraint dictionary key 'subset' must be a list of integers.")
            if constraint_dict["turbines"] != "existing":
                raise ValueError("Subset constraints only allowed for existing turbines.")
        else:
            constraint_dict["subset"] = None

        if self.verbose:
            print("Applying constraint: {0}".format(constraint_dict))

        if constraint_dict["name"] in self._constraint_masks_dict.keys():
            raise ValueError("Constraint name already in use.")

        if constraint_dict["turbines"] == "existing":
            if constraint_dict["value"] == "expected_power" or constraint_dict["value"] == "power":
                v = self.wake_map.process_existing_expected_powers(
                    subset=constraint_dict["subset"]
                )
            elif constraint_dict["value"] == "aep_loss":
                v = self.wake_map.process_existing_aep_loss(
                    subset=constraint_dict["subset"]
                )
            else:
                raise ValueError("Invalid value for constraint_dict['value']")

        elif constraint_dict["turbines"] == "candidates":
            if constraint_dict["value"] == "expected_power" or constraint_dict["value"] == "power":
                v = self.wake_map.process_candidate_expected_powers()
            elif constraint_dict["value"] == "aep_loss":
                v = self.wake_map.process_candidate_aep_loss()
            else:
                raise ValueError("Invalid value for constraint_dict['value']")

        else:
            raise ValueError("Invalid value for constraint_dict['turbines']")

        if constraint_dict["value"] == "aep_loss":
            self._constraint_masks_dict[constraint_dict["name"]] = v <= constraint_dict["threshold"]
        else:
            self._constraint_masks_dict[constraint_dict["name"]] = v >= constraint_dict["threshold"]

        self._constraints_list.append(constraint_dict)

        self._state = 1 # Constraints added, selection can proceed

    def reset_constraints(self):
        """
        Reset the constraints for the area selection process.

        Args:
            None
        Returns:
            None
        """
        self._constraint_masks_dict = {}
        self._constraints_list = []

    def report_constraints(self):
        """
        Print out the constraints that have been added.

        Args:
            None
        Returns:
            None (prints to console)
        """

        if self._constraint_masks_dict == {}:
            print("No constraints have been added.\n")
            return

        for n, m in self._constraint_masks_dict.items():
            print("Constraint: {0}".format(n))
            print("Turbine candidates meeting constraint: {0:.2f}%\n".format(sum(m)/len(m)*100))

    def add_objective(self, objective_dict: Dict[str, Any]):
        """
        Add an objective to the area selection process.
        Args:
            objective_dict: Dictionary containing the objective information.
                Must contain keys: value, candidates_weight, existing_weight, n_target
                value: "power", "capacity_factor", "normalized_power", or "aep_loss"
                candidates_weight: weight for candidate turbines
                existing_weight: weight for existing turbines
                n_target: number of turbines to select
        Returns:
            None
        """

        required_keys = ["value", "candidates_weight", "existing_weight", "n_target"]

        # Check that all inputs are valid
        if not all([key in objective_dict for key in required_keys]):
            raise ValueError("Objective dictionary must contain keys: {0}".format(required_keys))

        for key in ["candidates_weight", "existing_weight"]:
            if not isinstance(objective_dict[key], (int, float)):
                raise ValueError("Objective dictionary key '{0}' must be a number.".format(key))

        # TODO: handle subsets
        if "subset" in objective_dict.keys():
            raise NotImplementedError("Objectives for subsets have not yet been implemented.")

        self._objective_dict = objective_dict

        self._state = 1 # Objective added, selection can proceed

    def reset_objective(self):
        """
        Reset the objective(s) for the area selection process.

        Args:
            None
        Returns:
            None
        """

        self._objective_dict = {}

    def report_objective(self):
        """
        Print out the objectives that have been added.

        Args:
            None
        Returns:
            None (prints to console)
        """

        if self._objective_dict == {}:
            print("No objective has been added.\n")
            return

        print("Objective: {0}".format(self._objective_dict))

        return

    def select_candidates(self):
        """
        Select candidate locations based on previously-added constraints and objectives.

        Args:
            None
        Returns:
            None
        """
        if self._state != 1:
            raise RuntimeError(
                "Cannot select candidates until constraints or objective have been added."
            )

        # Get the mask for all constraints
        mask_all = np.ones_like(self._allowable_candidates_x, dtype=bool)
        for m in self._constraint_masks_dict.values():
            mask_all = mask_all & m

        self._selected_candidates_x = self._allowable_candidates_x[mask_all]
        self._selected_candidates_y = self._allowable_candidates_y[mask_all]

        if self.verbose:
            print("{:.2f}% of candidate locations remain after applying all constraints.\n".format(
                len(self._selected_candidates_x)/len(self._allowable_candidates_x)*100
            ))

        if self._objective_dict != {}:
            if self._objective_dict["n_target"] > len(self._selected_candidates_x):
                raise ValueError("Target number of turbines do not satisfy constraints.")

            if self._objective_dict["value"] == "power":
                v_c = self.wake_map.process_candidate_expected_powers()
                v_e = self.wake_map.process_existing_expected_powers()
            elif self._objective_dict["value"] == "capacity_factor":
                v_c = self.wake_map.process_candidate_expected_capacity_factors()
                v_e = self.wake_map.process_existing_expected_capacity_factors()
            elif self._objective_dict["value"] == "normalized_power":
                v_c = self.wake_map.process_candidate_expected_normalized_powers()
                v_e = self.wake_map.process_existing_expected_normalized_powers()
            elif self._objective_dict["value"] == "aep_loss":
                v_c = -self.wake_map.process_candidate_aep_loss()
                v_e = -self.wake_map.process_existing_aep_loss()

            v_both = (
                self._objective_dict["candidates_weight"]*v_c
                + self._objective_dict["existing_weight"]*v_e
            )

            self.objective_value = v_both.copy()
            v_both = v_both[mask_all]

            best_indices = np.argsort(v_both)[::-1][:self._objective_dict["n_target"]]

            self._selected_candidates_x = self._selected_candidates_x[best_indices]
            self._selected_candidates_y = self._selected_candidates_y[best_indices]

        self._state = 2 # Candidates selected

    def plot_selection(
        self,
        ax: plt.Axes | None = None,
        plotting_dict: Dict[str, Any] = {}
    ):
        """
        Plot the selected candidate locations.

        Args:
            ax: Axes object to plot on. If None, new axes will be created.
            plotting_dict: Dictionary of plotting options
        Returns:
            ax: Axes object with the selected candidate locations plotted.
        """

        if self._state != 2:
            raise RuntimeError("Cannot plot selection until candidates have been selected.")

        if ax is None:
            _, ax = plt.subplots()

        # Default to green for selected candidate locations
        if "color" not in plotting_dict.keys():
            plotting_dict["color"] = "green"

        fmodel_plot = self.wake_map.fmodel_all_candidates.copy()
        fmodel_plot.set(
            layout_x=self._selected_candidates_x,
            layout_y=self._selected_candidates_y
        )
        layout_viz.plot_turbine_points(fmodel_plot, ax=ax, plotting_dict=plotting_dict)

        return ax

    def plot_constraints(
        self,
        ax: plt.Axes | None = None,
        to_plot: list | None = None,
        plotting_dict: Dict[str, Any] = {}
    ):
        """
        UNDER DEVELOPMENT: Plot the constraints that have been added.

        Args:
            ax: Axes object to plot on. If None, new axes will be created.
            to_plot: List of constraint names to plot. If None, all constraints will be plotted.
            plotting_dict: Dictionary of plotting options
        Returns:
            ax: Axes object with the constraints plotted.
        """
        if self._state < 1:
            raise RuntimeError("Cannot plot constraints until constraints have been added.")

        if ax is None:
            _, ax = plt.subplots()

        #TODO more work here.
        """
        if constraint_dict["value"] == "aep_loss":
            self._constraint_masks_dict[constraint_dict["name"]] = v <= constraint_dict["threshold"]
        else:
            self._constraint_masks_dict[constraint_dict["name"]] = v >= constraint_dict["threshold"]
        """
        if to_plot is None:
            to_plot = list(self._constraint_masks_dict.keys())

        mask_all = np.ones_like(self._allowable_candidates_x, dtype=bool)
        for k in to_plot:
            mask_all = mask_all & self._constraint_masks_dict[k]


        ax.tricontourf(
            self.wake_map.all_candidates_x,
            self.wake_map.all_candidates_y,
            mask_all,
            vmin=0.5,
            vmax=1.5,
            colors="white",
            hatches=["/"]
        )

        return ax

    def plot_objective(
        self,
        ax: plt.Axes | None = None,
        cmap: str="viridis",
    ):
        """
        Plot the objective function value over the domain

        Args:
            ax: Axes object to plot on. If None, new axes will be created.
            cmap: Colormap to use for the objective function.
        Returns:
            ax: Axes object with the constraints plotted.
        """
        if self._state < 2:
            raise RuntimeError("Cannot plot objective until objective has been added.")

        if ax is None:
            _, ax = plt.subplots()

        self.wake_map.plot_contour(
            values = -self.objective_value,
            ax=ax,
            cmap=cmap,
            colorbar_label="Objective value"
        )

        return ax
