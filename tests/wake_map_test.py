import numpy as np
import pytest
from floris import FlorisModel, WindRose

from floriswakemap import WakeMap

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

wake_map_test = WakeMap(
    fmodel_test,
    wind_rose_test,
    min_dist=500,
    candidate_cluster_diameter=2000,
    boundaries=[(0, 0), (3000, 0), (3000, 3000), (0, 3000)],
    verbose=True
)

def test_instantiation():
    assert wake_map_test is not None

def test_certify_solved():
    wake_map_test._solved = True
    wake_map_test.certify_solved() # Should not raise error
    wake_map_test._solved = False
    with pytest.raises(AttributeError):
        wake_map_test.certify_solved()