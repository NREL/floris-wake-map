import numpy as np
from floris import FlorisModel, WindRose

from floriswakemap import WakeMap

wind_rose_test = WindRose(
    wind_speeds=np.array([8.0, 10.0]),
    wind_directions=np.array([45.0, 90.0, 135.0, 180.0, 225.0, 270.0]),
    freq_table=np.array([[0.2, 0.05], [0.2, 0.05], [0.0, 0.0], [0.0, 0.0], [0.37, 0.38], [0.5, 0.25]]),
    ti_table=0.06
)
wind_rose_test.plot()

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
    candidate_cluster_diameter=6000,
    boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)],
    verbose=True
)

wake_map.compute_raw_expected_powers_parallel()
print(
    "Shape of computed expected_powers_candidates:",
    wake_map.expected_powers_candidates_raw.shape
)
filename = "test_raw_expected_powers.npz"
wake_map.save_raw_expected_powers(filename)

wake_map.expected_powers_candidates_raw = None
wake_map.expected_powers_existing_raw = None

wake_map.load_raw_expected_powers(filename)
print(
    "Shape of loaded expected_powers_candidates:",
    wake_map.expected_powers_candidates_raw.shape
)
