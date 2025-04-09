import os

import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel, WindRose

from floriswakemap import WakeMap

if __name__ == "__main__":

    # Generate a wind rose for demonstration purposes
    wind_rose_demo = WindRose(
        wind_speeds=np.array([8.0, 10.0]),
        wind_directions=np.array([45.0, 90.0, 135.0, 180.0, 225.0, 270.0]),
        freq_table=np.array(
            [[0.2, 0.05], [0.2, 0.05], [0.0, 0.0], [0.0, 0.0], [0.37, 0.38], [0.5, 0.25]]
        ),
        ti_table=0.06
    )
    wind_rose_demo.plot()

    # Optionally, we can save the output figures
    save_figs = False
    if save_figs and not os.path.exists("figs"):
        os.makedirs("figs")

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

    # Establish the WakeMap object. Use a circular candidate cluster with a diameter of 6000 m.
    wake_map = WakeMap(
        fmodel,
        wind_rose_demo,
        min_dist=nautical_mile,
        boundaries=[(-10000, -10000), (25000, -10000), (25000, 25000), (-10000, 25000)],
        candidate_cluster_diameter=6000,
        verbose=True
    )

    # Plot the domain of the WakeMap object
    ax = wake_map.plot_existing_farm()
    if save_figs:
        fig = ax.get_figure()
        fig.savefig("figs/layouts_ex.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_locations(ax=ax)
    if save_figs:
        fig.savefig("figs/layouts_can.png", dpi=300, bbox_inches="tight", format="png")
    ax = wake_map.plot_candidate_layout(35, ax=ax)
    ax.legend()
    if save_figs:
        fig.savefig("figs/layouts_groups.png", dpi=300, bbox_inches="tight", format="png")

    # Run the main WakeMap computation process
    # (in serial, see also compute_raw_expected_powers_parallel)
    wake_map.compute_raw_expected_powers_serial()

    # Compute the expected powers of the existing and candidate groups
    existing_expected_powers = wake_map.process_existing_expected_powers()
    candidate_expected_powers = wake_map.process_candidate_expected_powers()

    # Print out the extracted outputs
    print("\nexisting_expected_powers.shape:", existing_expected_powers.shape)
    print("\ncandidate_expected_powers.shape:", candidate_expected_powers.shape)

    print("\nFirst five existing expected powers (W):\n", existing_expected_powers[:5])
    print("\nFirst five candidate expected powers (W):\n", candidate_expected_powers[:5])

    plt.show()
