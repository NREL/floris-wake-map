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
    min_dist=nm,
    group_diameter=6000,
    bounding_box={"x_min": -10000, "x_max": 25000, "y_min": -10000, "y_max": 25000},
    verbose=True
)

wakemap.compute_raw_expected_powers_parallel()
print(wakemap.expected_powers_candidates_raw.shape)
filename = "test_raw_expected_powers.npz"
wakemap.save_raw_expected_powers(filename)

wakemap.expected_powers_candidates_raw = None
wakemap.expected_powers_existing_raw = None

wakemap.load_raw_expected_powers(filename)

print(wakemap.expected_powers_candidates_raw.shape)
