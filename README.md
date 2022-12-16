[![PyPI version fury.io](https://badge.fury.io/py/invest-ucm-calibration.svg)](https://pypi.python.org/pypi/invest-ucm-calibration/)
[![Documentation Status](https://readthedocs.org/projects/invest-ucm-calibration/badge/?version=latest)](https://invest-ucm-calibration.readthedocs.io/en/latest/?badge=latest)
[![CI/CD](https://github.com/martibosch/invest-ucm-calibration/actions/workflows/dev.yml/badge.svg)](https://github.com/martibosch/invest-ucm-calibration/blob/main/.github/workflows/dev.yml)
[![codecov](https://codecov.io/gh/martibosch/invest-ucm-calibration/branch/main/graph/badge.svg)](https://codecov.io/gh/martibosch/invest-ucm-calibration)
[![GitHub license](https://img.shields.io/github/license/martibosch/invest-ucm-calibration.svg)](https://github.com/martibosch/invest-ucm-calibration/blob/master/LICENSE)

# InVEST urban cooling model calibration

## Overview

Automated calibration of the InVEST urban cooling model with simulated annealing

**Citation**: Bosch, M., Locatelli, M., Hamel, P., Remme, R. P., Chenal, J., and Joost, S. 2021. "A spatially-explicit approach to simulate urban heat mitigation with InVEST (v3.8.0)". *Geoscientific Model Development 14(6), 3521-3537*. [10.5194/gmd-14-3521-2021](https://doi.org/10.5194/gmd-14-3521-2021)

See [the user guide](https://invest-ucm-calibration.readthedocs.io/en/latest/user-guide.html) for more information, or [the `lausanne-heat-islands` repository](https://github.com/martibosch/lausanne-heat-islands) for an example use of this library in an academic article.

## Installation

This library requires the `gdal` library, which can easily be installed with conda as in:

```bash
conda install -c conda-forge gdal
```

Then, this library can be installed as in:

```bash
pip install invest-ucm-calibration
```

An alternative for the last step is to clone the repository and install it as in:

```bash
git clone https://github.com/martibosch/invest-ucm-calibration.git
python setup.py install
```

## TODO

- Allow a sequence of LULC rasters (although this would require an explicit mapping of each LULC/evapotranspiration/temperature raster or station measurement to a specific date)
- Test calibration based on `cc_method='intensity'`
- Support spatio-temporal datasets with [xarray](http://xarray.pydata.org) to avoid passing many separate rasters (and map each raster to a date more consistently)
- Read both station measurements and station locations as a single geo-data frame

## Acknowledgments

- The calibration procedure is based simulated annealing implementation of [perrygeo/simanneal](https://github.com/perrygeo/simanneal)
- With the support of the École Polytechnique Fédérale de Lausanne (EPFL)
- This package was created with the [ppw](https://zillionare.github.io/python-project-wizard) tool. For more information, please visit the [project page](https://zillionare.github.io/python-project-wizard/).
