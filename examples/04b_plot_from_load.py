import matplotlib.pyplot as plt
import numpy as np
import pickle

from floris import FlorisModel

from wakemap import WakeMap

if __name__ == "__main__":
    with open("/Users/msinner/cluster-wakes_JIP/cw_jip/analysis/wind_rose_vineyard.pkl", "rb") as f:
        wind_rose_vyw = pickle.load(f)
        wind_rose_vyw.heterogeneous_map = None
    wind_rose_vyw.plot()

    #wind_rose_vyw.downsample(wd_step=5.0, ws_step=2.0, inplace=True)
    #wind_rose_vyw.plot()
    fig = plt.gcf()
    fig.savefig("figs/vyw_windrose_full.png", dpi=300, bbox_inches="tight", format="png")


    fmodel = FlorisModel("gch.yaml")
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

    wakemap = WakeMap(
        fmodel,
        wind_rose_vyw,
        min_dist=nm,
        group_diameter=6000,
        boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)], # Changed! OK.
        external_losses_only=True,
        verbose=True
    )

    ax = wakemap.plot_existing_farm()
    fig = ax.get_figure()
    fig.savefig("figs/layouts_ex.png", dpi=300, bbox_inches="tight", format="png")
    ax = wakemap.plot_candidate_locations(ax=ax)
    fig.savefig("figs/layouts_can.png", dpi=300, bbox_inches="tight", format="png")
    ax = wakemap.plot_candidate_groups(35, ax=ax)
    fig.savefig("figs/layouts_groups.png", dpi=300, bbox_inches="tight", format="png")

    wakemap.load_raw_expected_powers("raw_expected_powers_vywr_full.npz")
    ee = wakemap.process_existing_expected_powers()
    ce = wakemap.process_candidate_expected_powers()

    # Candidate map (identical, as expected)
    ax = wakemap.plot_candidate_value(value="capacity_factor")
    ax = wakemap.plot_existing_farm(ax=ax)
    ax = wakemap.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    fig.savefig("figs/candidate_power_map_extonly_vywr_full.png", dpi=300, bbox_inches="tight", format="png")

    # Existing map (differ slightly in shape, magnitude shift. Unsurprising; seems reasonable)
    ax = wakemap.plot_existing_value(value="capacity_factor")
    ax = wakemap.plot_existing_farm(ax=ax)
    ax = wakemap.plot_candidate_locations(ax=ax)
    ax.set_aspect("equal")
    fig = ax.get_figure()
    fig.savefig("figs/existing_power_map_extonly_vywr_full.png", dpi=300, bbox_inches="tight", format="png")

    # Existing map, subset (as for full map).
    # Make a couple of different options here.
    subset_0 = [50, 51, 52, 53, 54, 60, 61, 62, 63, 64, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 90, 91, 92, 93, 94]
    subset_1 = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 40, 41, 42, 43, 44]
    subset_2 = [35, 36, 37, 45, 46, 47, 55, 56, 57, 65, 66, 67, 75, 76, 77, 85, 86, 87, 95, 96, 97]
    subset_3 = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27, 28, 29, 38, 39, 48, 49, 58, 59, 68, 69, 78, 79, 88, 89, 98, 99]
    subset_4 = subset_2 + subset_3
    subsets = [subset_0, subset_1, subset_2, subset_3, subset_4]
    for s, subset in enumerate(subsets):
        es = wakemap.process_existing_expected_capacity_factors_subset(subset=subset)
        ax = wakemap.plot_contour(
            es, cmap="Blues", colorbar_label="Subset turbine capacity factor [-]"
        )
        ax = wakemap.plot_existing_farm(ax=ax)
        ax = wakemap.plot_existing_farm(ax=ax, subset=subset, plotting_dict={"color": "red"})
        ax = wakemap.plot_candidate_locations(ax=ax)
        ax.set_aspect("equal")
        fig = ax.get_figure()
        fig.savefig("figs/subset_{0}_power_map_extonly_vywr_full.png".format(s), dpi=300, bbox_inches="tight", format="png")

    plt.show()
