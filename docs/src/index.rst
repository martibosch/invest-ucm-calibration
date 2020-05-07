InVEST urban cooling model calibration documentation
====================================================

Automated calibration of the InVEST urban cooling model with simulated annealing.

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
    
