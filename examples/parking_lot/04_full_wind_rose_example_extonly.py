"""
This example takes about 10 minutes to run locally.
"""
import pickle

import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel

from floriswakemap import WakeMap

if __name__ == "__main__":
    with open("../../cw_jip/analysis/wind_rose_vineyard.pkl", "rb") as f:
        wind_rose_vyw = pickle.load(f)
        wind_rose_vyw.heterogeneous_map = None
    wind_rose_vyw.plot()

    #wind_rose_vyw.downsample(wd_step=5.0, ws_step=2.0, inplace=True)
    #wind_rose_vyw.plot()
    fig = plt.gcf()
    fig.savefig("figs/vyw_windrose_full.png", dpi=300, bbox_inches="tight", format="png")


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
    print("Conditions in wind rose:", wind_rose_vyw.n_findex)
    print("Turbines in domain:", len(fmodel.layout_x))

    wake_map = WakeMap(
        fmodel,
        wind_rose_vyw,
        min_dist=nm,
        group_diameter=6000,
        boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)],
        external_losses_only=True,
        verbose=True
    )

    ax = wake_map.plot_existing_farm()
    fig = ax.get_figure()
    fig.savefig("figs/layouts_ex.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_locations(ax=ax)
    fig.savefig("figs/layouts_can.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_groups(35, ax=ax)
    fig.savefig("figs/layouts_groups.png", dpi=300, bbox_inches="tight", format="png")

    wake_map.compute_raw_expected_powers_serial()

    ee = wake_map.process_existing_expected_powers()
    ce = wake_map.process_candidate_expected_powers()

    # Candidate map (identical, as expected)
    ax = wake_map.plot_candidate_value(value="capacity_factor")
    ax = wake_map.plot_existing_farm(ax=ax)
    ax = wake_map.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    fig.savefig("figs/candidate_power_map_extonly_vywr_full.png", dpi=300, bbox_inches="tight", format="png")

    # Existing map (differ slightly in shape, magnitude shift. Unsurprising; seems reasonable)
    ax = wake_map.plot_existing_value(value="capacity_factor")
    ax = wake_map.plot_existing_farm(ax=ax)
    ax = wake_map.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    fig.savefig("figs/existing_power_map_extonly_vywr_full.png", dpi=300, bbox_inches="tight", format="png")

    # Existing map, subset (as for full map).
    # Make a couple of different options here.
    subset=range(10)
    es = wake_map.process_existing_expected_capacity_factors_subset(subset=subset)
    ax = wake_map.plot_contour(
        es, cmap="Blues", colorbar_label="Subset turbine capacity factor [-]"
    )
    ax = wake_map.plot_existing_farm(ax=ax)
    ax = wake_map.plot_existing_farm(ax=ax, subset=subset, plotting_dict={"color": "red"})
    ax = wake_map.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    fig.savefig("figs/subset_power_map_extonly_vywr_full.png", dpi=300, bbox_inches="tight", format="png")

    wake_map.save_raw_expected_powers("raw_expected_powers_vywr_full.npz")

    plt.show()
