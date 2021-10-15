InVEST urban cooling model calibration documentation
====================================================

Automated calibration of the InVEST urban cooling model with simulated annealing.

**Citation**: Bosch, M., Locatelli, M., Hamel, P., Remme, R. P., Chenal, J., and Joost, S. 2021. "A spatially-explicit approach to simulate urban heat mitigation with InVEST (v3.8.0)". *Geoscientific Model Development 14(6), 3521-3537*. `doi.org/10.5194/gmd-14-3521-2021 <https://doi.org/10.5194/gmd-14-3521-2021`_

.. toctree::
   :maxdepth: 1
   :caption: User guide:

   user-guide

.. toctree::
   :maxdepth: 1
   :caption: Reference guide:

   cli
   api
   
.. toctree::
   :maxdepth: 1
   :caption: Development:

   changelog
   contributing

Installation
------------

This library requires specific versions of the `gdal` and `rtree` libraries, which can easily be installed with conda as in:

.. code-block:: bash

    $ conda install -c conda-forge 'gdal<3.0' rtree 'shapely<1.7.0'

Then, this library can be installed as in:

.. code-block:: bash

    $ pip install invest-ucm-calibration

An alternative for the last step is to clone the repository and install it as in:

.. code-block:: bash

    $ git clone https://github.com/martibosch/invest-ucm-calibration.git
    $ python setup.py install
    
