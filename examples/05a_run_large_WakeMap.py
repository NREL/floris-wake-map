"""
This script generates the WakeMap object for example 05b. If called directly,
this script will execute the calculation of the wake map data, which takes
some time to run and is recommended for high-performance computing.
"""

import warnings
from time import perf_counter

import numpy as np
import pandas as pd
from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_gridded import (
    LayoutOptimizationGridded,
)

from floriswakemap import WakeMap


def construct_example_05_WakeMap():

    # Load and construct wind rose
    df = pd.read_csv("wind_rose_ex5.csv")
    ws = np.unique(df["ws"])
    wd = np.unique(df["wd"])
    freq = df["freq_val"].values.reshape((len(ws), len(wd))).T

    wind_rose = WindRose(
        wind_directions=wd,
        wind_speeds=ws,
        ti_table=0.06,
        freq_table=freq,
    )

    # Specify exclusion zones (3 separate zones)
    exclusion_zones = [
        np.array([
            [30, 40],
            [40, 55],
            [30, 60],
            [20, 35],
            [30, 40]
        ])*1e3,
        np.array([
            [20, 75],
            [40, 70],
            [45, 100],
            [20, 95],
            [20, 75]
        ])*1e3,
        np.array([
            [0, 112],
            [100, 120],
            [100, 122],
            [0, 114],
            [0, 112]
        ])*1e3,
    ]

    # Create some existing farms using FLORIS' LayoutOptimizationGridded class
    existing_boundary_1 = np.array([
        [50, 120],
        [80, 122],
        [45, 140],
        [40, 130],
        [50, 120],
    ])*1e3
    existing_boundary_2 = np.array([
        [80, 122],
        [91, 123],
        [100, 130],
        [102, 140],
        [50, 150],
        [45, 140],
        [80, 122],
    ])*1e3
    existing_boundary_3 = np.array([
        [50, 115],
        [90, 119],
        [70, 100],
        [66, 90],
        [45, 92],
        [50, 115],
    ])*1e3

    fmodel_1 = FlorisModel("defaults")
    fmodel_1.set(turbine_type=["iea_15MW"], wind_data=wind_rose, reference_wind_height=150.0)
    layout_opt_1 = LayoutOptimizationGridded(
        fmodel=fmodel_1,
        boundaries=[(x[0], x[1]) for x in existing_boundary_1],
        min_dist=1852,
        hexagonal_packing=True,
    )
    t_start = perf_counter()
    n_turbs_1, layout_x_1, layout_y_1 = layout_opt_1.optimize() # ~60 seconds
    print("Existing farm 1 layout created with {0} turbines in optimization time: {1:.1f} s".format(
        n_turbs_1, perf_counter() - t_start
    ))

    fmodel_2 = FlorisModel("defaults")
    fmodel_2.set(turbine_type=["iea_15MW"], wind_data=wind_rose, reference_wind_height=150.0)
    layout_opt_2 = LayoutOptimizationGridded(
        fmodel=fmodel_2,
        boundaries=[(x[0], x[1]) for x in existing_boundary_2],
        min_dist=1852,
    )
    t_start = perf_counter()
    n_turbs_2, layout_x_2, layout_y_2 = layout_opt_2.optimize() # ~120 seconds
    print("Existing farm 2 layout created with {0} turbines in optimization time: {1:.1f} s".format(
        n_turbs_2, perf_counter() - t_start
    ))

    fmodel_3 = FlorisModel("defaults")
    fmodel_3.set(turbine_type=["iea_15MW"], wind_data=wind_rose, reference_wind_height=150.0)
    layout_opt_3 = LayoutOptimizationGridded(
        fmodel=fmodel_3,
        boundaries=[(x[0], x[1]) for x in existing_boundary_3],
        min_dist=2000, # Slightly less dense
    )
    t_start = perf_counter()
    n_turbs_3, layout_x_3, layout_y_3 = layout_opt_3.optimize() # ~60 seconds
    print("Existing farm 3 layout created with {0} turbines in optimization time: {1:.1f} s".format(
        n_turbs_3, perf_counter() - t_start
    ))

    fmodel_1.set(layout_x=layout_x_1, layout_y=layout_y_1)
    fmodel_2.set(layout_x=layout_x_2, layout_y=layout_y_2)
    fmodel_3.set(layout_x=layout_x_3, layout_y=layout_y_3)

    fmodel_existing_all = FlorisModel.merge_floris_models([fmodel_1, fmodel_2, fmodel_3])

    # Switch to turboparkgauss with defaults
    fm_dict = fmodel_existing_all.core.as_dict()
    fm_dict["wake"] = {
        "model_strings": {
            "combination_model": "sosfs",
            "deflection_model": "none",
            "turbulence_model": "none",
            "velocity_model": "turboparkgauss"
         },
         "enable_secondary_steering": False,
         "enable_yaw_added_recovery": False,
         "enable_active_wake_mixing": False,
         "enable_transverse_velocities": False,
         "wake_velocity_parameters": {
            "turboparkgauss": {
                "A": 0.04,
                "include_mirror_wake": True,
            }
        },
        "wake_deflection_parameters": {
            "none": {}
        },
        "wake_turbulence_parameters": {
            "none": {}
        }
    }
    # fm_dict["solver"] = {
    #     "type": "turbine_cubature_grid",
    #     "turbine_grid_points": 4,
    # }
    fmodel_existing_all = FlorisModel(fm_dict)
    fmodel_existing_all.show_config()

    # Specify boundary for new lease area
    lease_boundary = np.array([
        [13, 50],
        [55, 50],
        [66, 90],
        [45, 92],
        [51, 118],
        [40, 130],
        [50, 150],
        [32, 150],
        [13, 50],
    ])*1e3

    # Instantiate WakeMap and return
    wake_map = WakeMap(
        fmodel_existing_all,
        wind_rose,
        min_dist=1852,
        boundaries=lease_boundary,
        candidate_turbine="iea_15MW",
        candidate_cluster_diameter=10000,
        exclusion_zones=exclusion_zones,
        verbose=True,
        silence_floris_warnings=True,
    )

    return wake_map

if __name__ == "__main__":
    # Create the WakeMap object, run the computation in parallel, and save the output
    wake_map = construct_example_05_WakeMap()
    filename = "example_05_raw_expected_powers.npz"
    with warnings.catch_warnings(action="ignore", category=RuntimeWarning):
        # Running with catch_warnings to quiet warning in gauss deflection model
        wake_map.compute_raw_expected_powers_parallel()
    wake_map.save_raw_expected_powers(filename)
    print("Script done")
