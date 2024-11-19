import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, WindRose

from wakemap import WakeMap


wind_rose_test = WindRose(
    wind_speeds=np.array([8.0, 10.0]),
    wind_directions=np.array([45.0, 90.0, 135.0, 180.0, 225.0, 270.0]),
    freq_table=np.array([[0.2, 0.05], [0.2, 0.05], [0.0, 0.0], [0.0, 0.0], [0.37, 0.38], [0.5, 0.25]]),
    ti_table=0.06
)
wind_rose_test.plot()

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

wakemap = WakeMap(
    fmodel,
    wind_rose_test,
    min_dist_D=nm/240,
    group_diameter_D=25,
    bounding_box={"x_min": -10000, "x_max": 25000, "y_min": -10000, "y_max": 25000},
    verbose=True
)

ax = wakemap.plot_existing_farm()
fig = ax.get_figure()
fig.savefig("figs/layouts_ex.png", dpi=300, bbox_inches="tight", format="png")
ax = wakemap.plot_candidate_locations(ax=ax)
fig.savefig("figs/layouts_can.png", dpi=300, bbox_inches="tight", format="png")
ax = wakemap.plot_candidate_groups(35, ax=ax)
fig.savefig("figs/layouts_groups.png", dpi=300, bbox_inches="tight", format="png")

wakemap.compute_raw_expected_powers_parallel()

ee = wakemap.process_existing_expected_powers()
ce = wakemap.process_candidate_expected_powers()

# Candidate map
ax = wakemap.plot_candidate_expected_powers(normalizer=1e6)
ax = wakemap.plot_existing_farm(ax=ax)
ax = wakemap.plot_candidate_locations(ax=ax)
fig = ax.get_figure()
fig.savefig("figs/candidate_power_map.png", dpi=300, bbox_inches="tight", format="png")

# Existing map
ax = wakemap.plot_existing_expected_powers(normalizer=1e6)
ax = wakemap.plot_existing_farm(ax=ax)
ax = wakemap.plot_candidate_locations(ax=ax)
fig = ax.get_figure()
fig.savefig("figs/existing_power_map.png", dpi=300, bbox_inches="tight", format="png")


plt.show()
