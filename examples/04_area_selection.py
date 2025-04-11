"""
NOTE: The AreaSelector class is in active development, and this example may
change.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel, WindRose

from floriswakemap import AreaSelector, WakeMap

if __name__ == "__main__":
    # Create demonstration wind rose
    wind_rose_demo = WindRose(
        wind_speeds=np.array([8.0, 10.0]),
        wind_directions=np.array([45.0, 90.0, 135.0, 180.0, 225.0, 270.0]),
        freq_table=np.array(
            [[0.2, 0.05], [0.2, 0.05], [0.0, 0.0], [0.0, 0.0], [0.37, 0.38], [0.5, 0.25]]
        ),
        ti_table=0.06
    )
    ax = wind_rose_demo.plot()

    # Optionally, we can save the output figures
    save_figs = True
    if save_figs and not os.path.exists("figs"):
        os.makedirs("figs")

    if save_figs:
        fig = ax.get_figure()
        fig.savefig("figs/ex04-1_wind_rose.png", dpi=300, bbox_inches="tight", format="png")

    # Set value to process and plot
    value = "aep_loss"

    # Instantiate a FlorisModel to represent the existing wind farm
    fmodel = FlorisModel("defaults")

    nautical_mile = 1852
    x_pos = np.linspace(0, 9*nautical_mile, 10)
    y_pos = x_pos
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)

    fmodel.set(
        layout_x=x_pos.flatten(),
        layout_y=y_pos.flatten(),
        turbine_type=["iea_15MW"],
        reference_wind_height=150.0
    )

    # For this example, we'll add some exclusion zones where candidate clusters cannot be placed.
    exclusion_zones = [
        [(15000, -7000), (20000, -2000), (15000, -2000)], # First zone
        [(-7000, 15000), (-4000, 15000), (-4000, 20000), (-7000, 20000)] # Second zone
    ]

    # Establish the WakeMap object. This time, we'll use a custom cluster layout defined by a set of
    # x, y coordinates. We'll also explicitly specify the candidate turbine type as the IEA 15MW
    # turbine, which comes with FLORIS. Alternatively, we could pass a FLORIS-compatible turbine
    # definition dictionary.
    candidate_cluster_layout=np.array([
        [0, 0],
        [0, 2000],
        [0, 4000],
        [2000, 0],
        [2000, 2000],
        [2000, 4000],
    ])
    wake_map = WakeMap(
        fmodel,
        wind_rose_demo,
        min_dist=nautical_mile,
        boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)],
        candidate_cluster_layout=candidate_cluster_layout,
        candidate_turbine="iea_15MW",
        exclusion_zones=exclusion_zones,
        verbose=True
    )

    # Plot the domain of the WakeMap object
    ax = wake_map.plot_existing_farm()
    ax = wake_map.plot_exclusion_zones(ax=ax)
    fig = ax.get_figure()
    if save_figs:
        fig.savefig("figs/ex04-2_existing.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_locations(ax=ax)
    if save_figs:
        fig.savefig("figs/ex04-3_candidate_locs.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_layout(35, ax=ax) # Note candidate cluster layout
    ax.legend()
    if save_figs:
        fig.savefig("figs/ex04-4_candidate_cluster.png", dpi=300, bbox_inches="tight", format="png")

    # Run the main WakeMap computation
    wake_map.compute_raw_expected_powers_parallel()

    # Instantiate the AreaSelector object.
    area_selector = AreaSelector(wake_map, verbose=True)

    # Add various constraints based on the existing and candidate wake maps.
    # Here, we're using aep_loss, the threshold applied will be taken as the
    # upper limit of allowable AEP loss.
    area_selector.report_constraints()
    area_selector.add_constraint({
        "turbines": "existing",
        "value": value,
        "threshold": 25,
        "name": "existing_AEP_loss"
    })
    area_selector.report_constraints()
    area_selector.add_constraint({
        "turbines": "candidates",
        "value": value,
        "threshold": 50,
        "name": "candidates_AEP_loss"
    })
    area_selector.report_constraints()

    # Select candidate locations based on added constraints (selects all candidates satisfying the
    # added constraints).
    area_selector.select_candidates()


    # Create plots showing the selected candidate locations on top of the wake maps.
    ax_c = wake_map.plot_candidate_value(value=value)
    ax_c.set_aspect("equal")
    ax_c = wake_map.plot_existing_farm(ax=ax_c)
    ax_c = wake_map.plot_candidate_locations(ax=ax_c)
    ax_c = wake_map.plot_exclusion_zones(ax=ax_c)
    ax_c = area_selector.plot_selection(ax=ax_c, plotting_dict={"label": "Meet all constraints"})
    fig_c = ax_c.get_figure()
    if save_figs:
        fig_c.savefig("figs/ex04-5_candidate_sel.png", dpi=300, bbox_inches="tight", format="png")

    # Same, but over the existing value map
    ax_e = wake_map.plot_existing_value(value=value)
    ax_e.set_aspect("equal")
    ax_e = wake_map.plot_existing_farm(ax=ax_e)
    ax_e = wake_map.plot_candidate_locations(ax=ax_e)
    ax_e = wake_map.plot_exclusion_zones(ax=ax_e)
    ax_e = area_selector.plot_selection(ax=ax_e, plotting_dict={"label": "Meet all constraints"})
    fig_e = ax_e.get_figure()
    if save_figs:
        fig_e.savefig("figs/ex04-6_existing_sel.png", dpi=300, bbox_inches="tight", format="png")

    # Add an objective of minimizing AEP loss. Equal weighting given to candidate and existing farm
    # objectives, and aiming to select 10 candidate locations that minimize the combined objective.
    area_selector.add_objective({
        "value": value,
        "candidates_weight": 0.5,
        "existing_weight": 0.5,
        "n_target": 10
    })

    # Rerun the selection procedure (applies constraints as before, but now uses objective)
    area_selector.select_candidates()

    # Overlay select on existing value map, showing selected candidate locations in blue
    ax_c = area_selector.plot_selection(
        ax=ax_c,
        plotting_dict={"color": "blue", "label": "Minimize AEP loss"}
    )
    ax_e = area_selector.plot_selection(
        ax=ax_e,
        plotting_dict={"color": "blue", "label": "Minimize AEP loss"}
    )
    if save_figs:
        fig_c.savefig("figs/ex04-7_candidate_opt.png", dpi=300, bbox_inches="tight", format="png")
        fig_e.savefig("figs/ex04-8_existing_opt.png", dpi=300, bbox_inches="tight", format="png")

    plt.show()
