[![PyPI version fury.io](https://badge.fury.io/py/invest-ucm-calibration.svg)](https://pypi.python.org/pypi/invest-ucm-calibration/)
[![Documentation Status](https://readthedocs.org/projects/invest-ucm-calibration/badge/?version=latest)](https://invest-ucm-calibration.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/martibosch/invest-ucm-calibration.svg?branch=master)](https://travis-ci.org/martibosch/invest-ucm-calibration)
[![Coverage Status](https://coveralls.io/repos/github/martibosch/invest-ucm-calibration/badge.svg?branch=master)](https://coveralls.io/github/martibosch/invest-ucm-calibration?branch=master)
[![GitHub license](https://img.shields.io/github/license/martibosch/invest-ucm-calibration.svg)](https://github.com/martibosch/invest-ucm-calibration/blob/master/LICENSE)

InVEST urban cooling model calibration
===============================

Overview
--------

Automated calibration of the InVEST urban cooling model with simulated annealing

See [the user guide](https://invest-ucm-calibration.readthedocs.io/en/latest/user-guide.html) for more information.

Installation
------------

This library requires specific versions of the `gdal` and `rtree` libraries, which can easily be installed with conda as in:

    $ conda install -c conda-forge 'gdal<3.0' rtree 'shapely<1.7.0'

Then, this library can be installed as in:

    $ pip install invest-ucm-calibration


An alternative for the last step is to clone the repository and install it as in:

    $ git clone https://github.com/martibosch/invest-ucm-calibration.git
    $ python setup.py install

TODO
----

* Allow a sequence of LULC rasters (although this would require an explicit mapping of each LULC/evapotranspiration/temperature raster or station measurement to a specific date)
* Test calibration based on `cc_method='intensity'`
* Support spatio-temporal datasets with [xarray](http://xarray.pydata.org) to avoid passing many separate rasters (and map each raster to a date more consistently)
* Read both station measurements and station locations as a single geo-data frame

