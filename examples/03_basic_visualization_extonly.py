import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, WindRose

from wakemap import WakeMap

if __name__ == "__main__":
    wind_rose_test = WindRose(
        wind_speeds=np.array([8.0, 10.0]),
        wind_directions=np.array([45.0, 90.0, 135.0, 180.0, 225.0, 270.0]),
        freq_table=np.array([[0.2, 0.05], [0.2, 0.05], [0.0, 0.0], [0.0, 0.0], [0.37, 0.38], [0.5, 0.25]]),
        ti_table=0.06
    )
    wind_rose_test.plot()

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

    wake_map = WakeMap(
        fmodel,
        wind_rose_test,
        min_dist=nm,
        group_diameter=6000,
        boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)],
        external_losses_only=True,
        verbose=True
    )

    ax = wake_map.plot_existing_farm()
    fig = ax.get_figure()
    if save_figs:
        fig.savefig("figs/layouts_ex.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_locations(ax=ax)
    if save_figs:
        fig.savefig("figs/layouts_can.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_groups(35, ax=ax)
    if save_figs:
        fig.savefig("figs/layouts_groups.png", dpi=300, bbox_inches="tight", format="png")

    wake_map.compute_raw_expected_powers_parallel()

    ee = wake_map.process_existing_expected_powers()
    ce = wake_map.process_candidate_expected_powers()

    # Candidate map (identical, as expected)
    ax = wake_map.plot_candidate_value(value=value)
    ax = wake_map.plot_existing_farm(ax=ax)
    ax = wake_map.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    if save_figs:
        fig.savefig("figs/candidate_power_map_extonly.png", dpi=300, bbox_inches="tight", format="png")

    # Existing map (differ slightly in shape, magnitude shift. Unsurprising; seems reasonable)
    ax = wake_map.plot_existing_value(value=value)
    ax = wake_map.plot_existing_farm(ax=ax)
    ax = wake_map.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    if save_figs:
        fig.savefig("figs/existing_power_map_extonly.png", dpi=300, bbox_inches="tight", format="png")

    # Existing map, subset (as for full map).
    subset=range(10)
    es = wake_map.process_existing_expected_normalized_powers()
    cs = wake_map.process_candidate_expected_normalized_powers()
    es = wake_map.process_existing_expected_capacity_factors_subset(subset=subset)
    ax = wake_map.plot_contour(
        es, cmap="Blues", colorbar_label="Subset turbine capacity factor [-]"
    )
    ax = wake_map.plot_existing_farm(ax=ax)
    ax = wake_map.plot_existing_farm(ax=ax, subset=subset, plotting_dict={"color": "red"})
    ax = wake_map.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    if save_figs:
        fig.savefig("figs/subset_power_map_extonly.png", dpi=300, bbox_inches="tight", format="png")

    plt.show()
