import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, WindRose

from floriswakemap import WakeMap, AreaSelector

if __name__ == "__main__":
    wind_rose_test = WindRose(
        wind_speeds=np.array([8.0, 10.0]),
        wind_directions=np.array([45.0, 90.0, 135.0, 180.0, 225.0, 270.0]),
        freq_table=np.array([[0.2, 0.05], [0.2, 0.05], [0.0, 0.0], [0.0, 0.0], [0.37, 0.38], [0.5, 0.25]]),
        ti_table=0.06
    )

    save_figs = False

    value = "capacity_factor"

    fmodel = FlorisModel("inputs/gch.yaml")
    fmodel.set(turbine_type=["iea_15MW"], reference_wind_height=150.0)
    nm = 1852
    x_pos = np.linspace(0, 9*nm, 10)
    y_pos = x_pos
    x_pos, y_pos = np.meshgrid(x_pos, y_pos)

    fmodel.set(
        layout_x=x_pos.flatten(),
        layout_y=y_pos.flatten(),
    )

    exclusion_zone = [(15000, -5000), (20000, -2000), (15000, -2000)]

    wake_map = WakeMap(
        fmodel,
        wind_rose_test,
        min_dist=nm,
        group_diameter=6000,
        boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)],
        exclusion_zones=[exclusion_zone],
        external_losses_only=True,
        verbose=True
    )

    wake_map.compute_raw_expected_powers_parallel()

    # Create the area selector and add constraints to it.
    area_selector = AreaSelector(wake_map, verbose=True)

    # I think I'll need something more "normalized"---a capacity factor loss or similar.
    area_selector.report_constraints()
    area_selector.add_constraint({
        "turbines": "existing",
        "value": value,
        "threshold": 0.561,
        "name": "existing_CF"
    })
    area_selector.report_constraints()
    area_selector.add_constraint({
        "turbines": "candidates",
        "value": value,
        "threshold": 0.530,
        "name": "candidates_CF"
    })
    area_selector.report_constraints()

    # Apply constraints and select candidates.
    area_selector.select_candidates()


    # Make some nice plots
    # Candidate map
    ax_c = wake_map.plot_candidate_value(value=value)
    ax_c.set_aspect("equal")
    ax_c = wake_map.plot_existing_farm(ax=ax_c)
    ax_c = wake_map.plot_candidate_locations(ax=ax_c)
    ax_c = wake_map.plot_exclusion_zones(ax=ax_c)
    ax_c = area_selector.plot_selection(ax=ax_c)
    
    # Existing map (differ slightly in shape, magnitude shift. Unsurprising; seems reasonable)
    ax_e = wake_map.plot_existing_value(value=value)
    ax_e.set_aspect("equal")
    ax_e = wake_map.plot_existing_farm(ax=ax_e)
    ax_e = wake_map.plot_candidate_locations(ax=ax_e)
    ax_e = wake_map.plot_exclusion_zones(ax=ax_e)
    ax_e = area_selector.plot_selection(ax=ax_e)

    area_selector.add_objective({
        "value": value,
        "candidates_weight": 0.5,
        "existing_weight": 0.5,
        "n_target": 10
    })

    area_selector.select_candidates()
    ax_c = area_selector.plot_selection(ax=ax_c, plotting_dict={"color": "blue"})
    ax_e = area_selector.plot_selection(ax=ax_e, plotting_dict={"color": "blue"})


    plt.show()
