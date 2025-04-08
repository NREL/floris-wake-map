import copy
import os

import numpy as np
import pytest
from floris import FlorisModel, WindRose

from floriswakemap import WakeMap
from floriswakemap.wake_map import (
    _compute_expected_powers_existing_single_external_only,
    _sample_locations,
)

# Set up test data
wind_rose_test = WindRose(
    wind_speeds=np.array([8.0, 10.0]),
    wind_directions=np.array([45.0, 90.0]),
    freq_table=np.array(
        [[0.2, 0.05], [0.2, 0.05]]
    ),
    ti_table=0.06
)

fmodel_test = FlorisModel("defaults")
fmodel_test.set(layout_x=[0, 0, 500, 500], layout_y=[0, 500, 0, 500])

wm_test = WakeMap(
    fmodel_test,
    wind_rose_test,
    min_dist=500,
    boundaries=[(-1500, -1500), (1500, -1500), (1500, 1500), (-1500, 1500)],
    candidate_cluster_diameter=2000,
    verbose=True,
    parallel_max_workers=2
)

base_expected_powers_candidates_raw_44 = np.array(
    [
        [6652247.53867076, 7595558.61856194, 6658264.86724991, 6332515.89426748],
        [7595558.61856194, 7595804.62252654, 6332515.89426748, 6163129.52751615],
        [7595804.62252654, 7595804.62252654, 6163129.52751615, 7595802.97760993],
        [7595804.62252654, 7595804.62252654, 7595802.97760993, 7595804.62252654],
    ]
)

base_expected_powers_existing_raw_44 = np.array(
    [
        [2086722.96836165, 2086722.96836165, 2086722.96836165, 2086722.96836165],
        [2086722.96836165, 2086722.96836165, 2086722.96836165, 2086722.96836165],
        [2070809.99862224, 2086722.96836165, 2086722.96836165, 2086722.96836165],
        [2057647.87009672, 2086722.96836165, 2070809.99862224, 2086722.96836165]
    ]
)

def test_instantiation():
    assert wm_test is not None

def test_certify_solved():
    wm_test._solved = True
    wm_test.certify_solved() # Should not raise error
    wm_test._solved = False
    with pytest.raises(AttributeError):
        wm_test.certify_solved()

def test_create_candidate_locations():
    # WakeMap.create_candidate_locations() is called during instantiation;
    # this test just checks that the boundaries and minimum distance constraints
    # are being satisfied by the locations.

    # Check boundaries
    assert (wm_test.all_candidates_x >= -1500).all()
    assert (wm_test.all_candidates_x <= 1500).all()
    assert (wm_test.all_candidates_y >= -1500).all()
    assert (wm_test.all_candidates_y <= 1500).all()

    # Check minimum distance from other candidates
    for i in range(wm_test.n_candidates):
        for j in range(i+1, wm_test.n_candidates):
            assert np.sqrt(
                (wm_test.all_candidates_x[i] - wm_test.all_candidates_x[j])**2 +
                (wm_test.all_candidates_y[i] - wm_test.all_candidates_y[j])**2
            ) >= 500

    # Check minimum distance for existing locations
    for i in range(wm_test.n_candidates):
        for j in range(len(wm_test.fmodel_existing.layout_x)):
            assert np.sqrt(
                (wm_test.all_candidates_x[i] - wm_test.fmodel_existing.layout_x[j])**2 +
                (wm_test.all_candidates_y[i] - wm_test.fmodel_existing.layout_y[j])**2
            ) >= 500

def test_create_candidate_clusters():
    wm_test_2 = copy.deepcopy(wm_test)

    # Check that a smaller diameter results in a smaller cluster
    wm_test_2.create_candidate_clusters(None, 1000)
    assert wm_test_2.candidate_layout.shape[1] == 2
    assert wm_test_2.candidate_layout.shape[0] < wm_test.candidate_layout.shape[0]

    # Check that a specified layout results in a different cluster, and that
    # candidate_cluster_diameter is ignored
    n = wm_test.candidate_layout.shape[0]
    layout_x = np.linspace(0, n*500, n)
    layout_y = np.zeros_like(layout_x)
    wm_test_2.create_candidate_clusters(np.column_stack((layout_x, layout_y)), 0)
    assert np.allclose(wm_test_2.candidate_layout.mean(axis=0), 0)
    assert np.allclose(wm_test_2.candidate_layout[:,0], layout_x - layout_x.mean())
    assert np.allclose(wm_test_2.candidate_layout[:,1], layout_y - layout_y.mean())
    assert wm_test_2.candidate_layout.shape[0] == wm_test.candidate_layout.shape[0]

def test_compute_raw_expected_powers_serial():
    wm_test.compute_raw_expected_powers_serial()

    # Check shapes
    assert (np.array(wm_test.expected_powers_existing_raw).shape
            == (wm_test.n_candidates, len(wm_test.fmodel_existing.layout_x)))
    assert (wm_test.expected_powers_candidates_raw.shape
            == (wm_test.n_candidates, wm_test.candidate_layout.shape[0]))

    # Regression tests for power
    assert np.allclose(
        wm_test.expected_powers_candidates_raw[:4,:4],
        base_expected_powers_candidates_raw_44
    )
    assert np.allclose(
        np.array(wm_test.expected_powers_existing_raw)[:4,:4],
        base_expected_powers_existing_raw_44
    )

def test_compute_raw_expected_powers_parallel():
    wm_test.compute_raw_expected_powers_parallel()

    # Check shapes
    assert (np.array(wm_test.expected_powers_existing_raw).shape
            == (wm_test.n_candidates, len(wm_test.fmodel_existing.layout_x)))
    assert (wm_test.expected_powers_candidates_raw.shape
            == (wm_test.n_candidates, wm_test.candidate_layout.shape[0]))

    # Regression tests for power
    assert np.allclose(
        wm_test.expected_powers_candidates_raw[:4,:4],
        base_expected_powers_candidates_raw_44
    )
    assert np.allclose(
        np.array(wm_test.expected_powers_existing_raw)[:4,:4],
        base_expected_powers_existing_raw_44
    )

def test_compute_expected_powers_candidates():
    # Already compute in previous tests, but this will check test_compute_expected_powers_candidates
    # specifically.

    wm_test.compute_expected_powers_candidates()
    assert (wm_test.expected_powers_candidates_raw.shape
            == (wm_test.n_candidates, wm_test.candidate_layout.shape[0]))
    assert np.allclose(
        wm_test.expected_powers_candidates_raw[:4,:4],
        base_expected_powers_candidates_raw_44
    )

def test_process_existing_expected_powers():
    out = wm_test.process_existing_expected_powers()

    assert out.shape[0] == wm_test.n_candidates
    assert np.allclose(out, np.mean(wm_test.expected_powers_existing_raw, axis=1))

    # Test on a subset
    subset = [2,3]
    out = wm_test.process_existing_expected_powers(subset)

    assert out.shape[0] == wm_test.n_candidates
    assert np.allclose(
        out,
        np.mean(np.array(wm_test.expected_powers_existing_raw)[:,subset], axis=1)
    )

def test_process_candidate_expected_powers():
    out = wm_test.process_candidate_expected_powers()

    assert out.shape[0] == wm_test.n_candidates
    assert np.allclose(out, np.mean(wm_test.expected_powers_candidates_raw, axis=1))

def test_process_existing_aep_loss():
    hours_per_year_default = 8760
    out = wm_test.process_existing_aep_loss()
    assert out.shape[0] == wm_test.n_candidates

    wm_test.fmodel_existing.run_no_wake()
    unwaked_powers = wm_test.fmodel_existing.get_expected_turbine_powers()
    expected_power_loss = np.sum(
        unwaked_powers.reshape(1, wm_test.fmodel_existing.n_turbines)
        - wm_test.expected_powers_existing_raw,
        axis = 1
    ) / 1e9
    assert np.allclose(
        out,
        expected_power_loss * hours_per_year_default
    )

    # Test with a different number of hours
    hours_per_year = 1000
    out = wm_test.process_existing_aep_loss(hours_per_year=hours_per_year)
    assert out.shape[0] == wm_test.n_candidates
    assert np.allclose(
        out,
        expected_power_loss * hours_per_year
    )

    # Test a subset
    subset = [2,3]
    out = wm_test.process_existing_aep_loss(subset)
    assert out.shape[0] == wm_test.n_candidates

    wm_test.fmodel_existing.run_no_wake()
    unwaked_powers = wm_test.fmodel_existing.get_expected_turbine_powers()[subset]
    expected_power_loss = np.sum(
        unwaked_powers.reshape(1, len(subset))
        - np.array(wm_test.expected_powers_existing_raw)[:,subset],
        axis = 1
    ) / 1e9
    assert np.allclose(
        out,
        expected_power_loss * hours_per_year_default
    )

    hours_per_year = 1000
    out = wm_test.process_existing_aep_loss(subset, hours_per_year=hours_per_year)
    assert out.shape[0] == wm_test.n_candidates
    assert np.allclose(
        out,
        expected_power_loss * hours_per_year
    )

def test_process_candidate_aep_loss():
    hours_per_year_default = 8760
    out = wm_test.process_candidate_aep_loss()
    assert out.shape[0] == wm_test.n_candidates

    wm_test.fmodel_all_candidates.run_no_wake()
    unwaked_powers = wm_test.fmodel_all_candidates.get_expected_turbine_powers()
    expected_power_loss = np.sum(
        unwaked_powers.reshape(wm_test.n_candidates, 1)
        - wm_test.expected_powers_candidates_raw,
        axis = 1
    ) / 1e9
    assert np.allclose(
        out,
        expected_power_loss * hours_per_year_default
    )

    # Test with a different number of hours
    hours_per_year = 1000
    out = wm_test.process_candidate_aep_loss(hours_per_year=hours_per_year)
    assert out.shape[0] == wm_test.n_candidates
    assert np.allclose(
        out,
        expected_power_loss * hours_per_year
    )

def test_plotting_integration():
    # Won't be checking the outputs of these; just checking that they run without error.
    value_options = ["expected_power", "aep_loss"]

    # Layout plots
    ax = wm_test.plot_existing_farm()
    ax = wm_test.plot_candidate_locations(ax=ax)
    with pytest.raises(TypeError):
        wm_test.plot_candidate_layout(ax=ax) # Missing required argument candidate_idx
    ax = wm_test.plot_candidate_layout(candidate_idx=0, ax=ax)
    ax = wm_test.plot_exclusion_zones()
    ax = wm_test.plot_candidate_boundary()

    # Low-level contour plot
    with pytest.raises(TypeError):
        wm_test.plot_contour() # Missing required argument values
    ax = wm_test.plot_contour(values=wm_test.process_existing_expected_powers())

    # High-level contour plots
    for v in value_options:
        ax = wm_test.plot_existing_value(v)
        ax = wm_test.plot_candidate_value(v)
        ax = wm_test.plot_existing_value(v, subset=[2,3])

def test_save_and_load():
    wm_test.save_raw_expected_powers("test_save.npz")
    wm_test.expected_powers_candidates_raw = None
    wm_test.expected_powers_existing_raw = None
    wm_test.load_raw_expected_powers("test_save.npz")
    assert np.allclose(
        wm_test.expected_powers_candidates_raw[:4,:4],
        base_expected_powers_candidates_raw_44
    )
    assert np.allclose(
        np.array(wm_test.expected_powers_existing_raw)[:4,:4],
        base_expected_powers_existing_raw_44
    )
    os.remove("test_save.npz")

# Helper functions
def test_compute_expected_powers_existing_single_external_only():
    idx_test = 2
    out = _compute_expected_powers_existing_single_external_only(
        wm_test.fmodel_existing,
        wm_test.fmodel_all_candidates,
        wm_test.candidate_layout,
        idx_test
    )

    assert np.allclose(out, base_expected_powers_existing_raw_44[idx_test,:])

def test_sample_locations():
    # Run sample_locations and extract x, y, z points
    x, y, z = _sample_locations(wm_test.fmodel_existing)

    # Check they agree with fmodel_existing
    hub_heights = [t["hub_height"] for t in wm_test.fmodel_existing.core.farm.turbine_definitions]
    assert np.allclose(np.mean(x, axis=(1,2)), wm_test.fmodel_existing.layout_x)
    assert np.allclose(np.mean(y, axis=(1,2)), wm_test.fmodel_existing.layout_y)
    assert np.allclose(np.mean(z, axis=(1,2)), hub_heights)
