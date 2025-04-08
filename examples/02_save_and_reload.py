import numpy as np
from floris import FlorisModel, WindRose

from floriswakemap import WakeMap

# Create demonstration wind rose
wind_rose_demo = WindRose(
    wind_speeds=np.array([8.0, 10.0]),
    wind_directions=np.array([45.0, 90.0, 135.0, 180.0, 225.0, 270.0]),
    freq_table=np.array(
        [[0.2, 0.05], [0.2, 0.05], [0.0, 0.0], [0.0, 0.0], [0.37, 0.38], [0.5, 0.25]]
    ),
    ti_table=0.06
)

# Instantiate a FlorisModel to represent the existing wind farn
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

# Establish WakeMap object
wake_map = WakeMap(
    fmodel,
    wind_rose_demo,
    min_dist=nautical_mile,
    boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)],
    candidate_cluster_diameter=6000,
    verbose=True
)

# Run main computation in parallel and report output shape
wake_map.compute_raw_expected_powers_parallel()
print(
    "Shape of computed expected_powers_candidates_raw:",
    wake_map.expected_powers_candidates_raw.shape
)

# Save the raw data
filename = "saved_raw_expected_powers.npz"
wake_map.save_raw_expected_powers(filename)

# Overwrite the raw data on the WakeMap object to demonstrate that it will be
# reloaded from file
wake_map.expected_powers_candidates_raw = None
wake_map.expected_powers_existing_raw = None

# Load the raw data from file
wake_map.load_raw_expected_powers(filename)
print(
    "Shape of loaded expected_powers_candidates:",
    wake_map.expected_powers_candidates_raw.shape
)
