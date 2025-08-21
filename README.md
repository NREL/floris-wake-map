# FLORIS Wake Map

The floris-wake-map repository is a set of tools for creating maps of cluster
wake impacts based on the wind farm modeling tool [FLORIS](https://github.com/nrel/floris).

The `floriswakemap` package comprises two main modules: the `wake_map` module, dedicated to the
`WakeMap` class, and the `area_selector` module, dedicated to the `AreaSelector` class.
The `WakeMap` class is used to generate maps of wake impacts on both new and existing wind farm
clusters, while the `AreaSelector` class provides selections of candidate areas for new developments
based on an instantiated `WakeMap` object.

## Installation

To use the `floriswakemap` package, first clone the repository onto your computer by running the
following from your shell:
```
git clone https://github.com/nrel/floris-wake-map
```

We recommend that you install `floriswakemap` into a conda environment. To create a conda
environment and activate it, use
```
conda create --name my-environment python
conda activate my-environment
```
(where you may swap `my-environment` for your desired environment name).

Then, perform a local pip installation of the `floriswakemap` repository.
```
cd floris-wake-map
pip install -e .
```

You can verify that `floriswakemap` has installed by running
```
pip show floriswakemap
```
and confirming that the output provides information about the package.

## Getting started

To get familiar with the `floriswakemap`, we recommend working through the examples (in the
examples/ subdirectory) in numerical order. In particular:
- 01_basic_usage.py demonstrates the main workflow for using `WakeMap` with an instantiated
`FlorisModel` and how to generate visualizations.
- 02_save_and_reload.py demonstrates the process of saving computed maps and reloading them, since
generating the wake maps is the most computationally demanding step.
- 03_exclusions_zones.py shows how to add more complex regions to the wake map.
- 04_area_selection.py shows how the `WakeMap` object can be used with the `AreaSelector` class to
select feasible and least-cost development areas.
- 05_full_example.py demonstrates many features in a larger problem, which takes a while to run and
should be run only once users are comfortable with the preceding examples.

Note that under the hood, `WakeMap` calculations using `compute_raw_expected_powers_parallel()`
(the main simulation method) are designed to run in parallel across multiple cores. To run
serially, instead use `compute_raw_expected_powers_serial()`. This will be considerably slower
to run, depending on the availability of parallel computational resources.
