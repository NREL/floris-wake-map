from typing import (
    Any,
    Dict,
)

import numpy as np
import matplotlib.pyplot as plt

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

    def add_constraint(self, constraint_dict):

        required_keys = ["turbines", "value", "threshold", "name"]

        # Check that all inputs are valid
        if not all([key in constraint_dict for key in required_keys]):
            raise ValueError("Constraint dictionary must contain keys: {0}".format(required_keys))
        
        if constraint_dict["turbines"] not in ["existing", "candidates"]:
            raise ValueError(
                "Constraint dictionary key 'turbines' must be either 'existing' or 'candidate'."
            )
        
        #if constraint_dict["value"] not in ["power", "AEP"]:
        #    raise ValueError("Constraint dictionary key 'value' must be either 'power' or 'AEP'.")

        if "subset" in constraint_dict.keys():
            if not isinstance(constraint_dict["subset"], list):
                raise ValueError("Constraint dictionary key 'subset' must be a list of integers.")
            if constraint_dict["turbines"] != "existing":
                raise ValueError("Subset constraints only allowed for existing turbines.")

        if self.verbose:
            print("Applying constraint: {0}".format(constraint_dict))

        if constraint_dict["name"] in self._constraint_masks_dict.keys():
            raise ValueError("Constraint name already in use.")

        # See about better handling here, if possible.
        
        if constraint_dict["turbines"] == "existing":
            if "subset" in constraint_dict.keys():
                if constraint_dict["value"] == "power":
                    v = self.wake_map.process_existing_expected_powers_subset(
                        subset=constraint_dict["subset"]
                    )
                elif constraint_dict["value"] == "capacity_factor":
                    v = self.wake_map.process_existing_expected_capacity_factors_subset(
                        subset=constraint_dict["subset"]
                    )
                elif constraint_dict["value"] == "normalized_power":
                    v = self.wake_map.process_existing_expected_normalized_powers_subset(
                        subset=constraint_dict["subset"]
                    )
                elif constraint_dict["value"] == "aep_loss":
                    v = self.wake_map.process_existing_aep_loss_subset(
                        subset=constraint_dict["subset"]
                    )
                else:
                    # Should already have checked, so this shouldn't happen.
                    raise ValueError("Invalid value for constraint_dict['value']")
            else:
                if constraint_dict["value"] == "power":
                    v = self.wake_map.process_existing_expected_powers()
                elif constraint_dict["value"] == "capacity_factor":
                    v = self.wake_map.process_existing_expected_capacity_factors()
                elif constraint_dict["value"] == "normalized_power":
                    v = self.wake_map.process_existing_expected_normalized_powers()
                elif constraint_dict["value"] == "aep_loss":
                    v = self.wake_map.process_existing_aep_loss()
                else:
                    # Should already have checked, so this shouldn't happen.
                    raise ValueError("Invalid value for constraint_dict['value']")
        elif constraint_dict["turbines"] == "candidates":
            if constraint_dict["value"] == "power":
                v = self.wake_map.process_candidate_expected_powers()
            elif constraint_dict["value"] == "capacity_factor":
                v = self.wake_map.process_candidate_expected_capacity_factors()
            elif constraint_dict["value"] == "normalized_power":
                v = self.wake_map.process_candidate_expected_normalized_powers()
            elif constraint_dict["value"] == "aep_loss":
                v = self.wake_map.process_candidate_aep_loss()
            else:
                # Should already have checked, so this shouldn't happen.
                raise ValueError("Invalid value for constraint_dict['value']")
        else:
            # Should already have checked, so this shouldn't happen.
            raise ValueError("Invalid value for constraint_dict['turbines']")

        if constraint_dict["value"] == "aep_loss":
            self._constraint_masks_dict[constraint_dict["name"]] = v <= constraint_dict["threshold"]
        else:
            self._constraint_masks_dict[constraint_dict["name"]] = v >= constraint_dict["threshold"]

        self._state = 1 # Constraints added, selection can proceed

    def reset_constraints(self):
        self._constraint_masks_dict = {}

    def report_constraints(self):

        if self._constraint_masks_dict == {}:
            print("No constraints have been added.\n")
            return

        for n, m in self._constraint_masks_dict.items():
            print("Constraint: {0}".format(n))
            print("Turbine candidates meeting constraint: {0:.2f}%\n".format(sum(m)/len(m)*100))

        return
    
    def add_objective(self, objective_dict):
        # TODO: add this in
        self._state = 1 # Objective added, selection can proceed

        required_keys = ["value", "candidates_weight", "existing_weight", "n_target"]

        # Check that all inputs are valid
        if not all([key in objective_dict for key in required_keys]):
            raise ValueError("Objective dictionary must contain keys: {0}".format(required_keys))
        
        for key in ["candidates_weight", "existing_weight"]:
            if not isinstance(objective_dict[key], (int, float)):
                raise ValueError("Objective dictionary key '{0}' must be a number.".format(key))
            
        # TODO: handle subsets
        self._objective_dict = objective_dict

        self._state = 1 # Objective added, selection can proceed

    def reset_objective(self):
        self._objective_dict = {}

    def report_objective(self):
        if self._objective_dict == {}:
            print("No objective has been added.\n")
            return

        print("Objective: {0}".format(self._objective_dict))

        return

    def select_candidates(self):
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

        if self._state != 2:
            raise RuntimeError("Cannot plot selection until candidates have been selected.")
        
        if ax is None:
            _, ax = plt.subplots()
    
        # ax = self.wake_map.plot_existing_farm(ax=ax)
        # ax = self.wake_map.plot_candidate_locations(ax=ax)
        # ax = self.wake_map.plot_exclusion_zones(ax=ax)

        # Green for selected candidate locations
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
        plotting_dict: Dict[str, Any] = {}
    ):
        if self._state < 1:
            raise RuntimeError("Cannot plot constraints until constraints have been added.")

        if ax is None:
            _, ax = plt.subplots()

        #TODO more work here.
        import ipdb; ipdb.set_trace()
        if constraint_dict["value"] == "aep_loss":
            self._constraint_masks_dict[constraint_dict["name"]] = v <= constraint_dict["threshold"]
        else:
            self._constraint_masks_dict[constraint_dict["name"]] = v >= constraint_dict["threshold"]

        
        
        ctrf = ax.tricontourf(
            self.all_candidates_x,
            self.all_candidates_y,
            values/normalizer,
            cmap=cmap
        )

