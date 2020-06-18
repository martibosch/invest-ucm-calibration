==========
User guide
==========

------
Set up
------

First of all, install the `invest-ucm-calibration` library following the instructions of the index of this documentation.


Then, in order to have some data that we can work on, let us first clone the repository:

.. code-block:: bash

    $ git clone https://github.com/martibosch/invest-ucm-calibration.git

We will be using the data from the tests, so let us move to that directory:

.. code-block:: bash

    $ cd invest-ucm-calibration/tests/data

Note that we have temperature and reference evapotranspiration data for two days, noted respectively with a 0 and 1 at the end of the file names. Also note that since we are using toy data, the number of calibration steps will be set to 10 in all the examples below (and the number of updates for which we log the process is also set to 10, so that we log every calibration step). Nonetheless, proper calibration in real settings will most likely require more calibration steps (the default is set to 100 calibration steps).
    
----------------------------------
Calibration with a temperature map
----------------------------------

We can calibrate the model to best-fit the temperature map of `T0.tif` as in:

.. code-block:: bash

    $ invest-ucm-calibration lulc.tif biophysical-table.csv factors \
                --ref-et-raster-filepaths ref_et0.tif --t-raster-filepaths T0.tif \
                --num-steps 10 --num-update-logs 10 \
                --dst-filepath calibrated-params.json

which will dump the calibrated parameters to `calibrated-params.json`.

-------------------------------------
Calibration with station measurements
-------------------------------------

We can calibrate the model to best-fit the station measurements of `station-t-one-day.csv` (with the station locations from `station-locations.csv`) as in:

.. code-block:: bash

    $ invest-ucm-calibration lulc.tif biophysical-table.csv factors \
                --ref-et-raster-filepaths ref_et0.tif \
                --station-t-filepath station-t-one-day.csv \
                --station-locations station-locations.csv --num-steps 10 \
                --num-update-logs 10 --dst-filepath calibrated-params.json

which will dump the calibrated parameters to `calibrated-params.json`.

-----------------------------
Calibration for multiple days
-----------------------------

Regardless of whether we are calibrating with a temperare map or station measurements, we can calibratethe model to best-fit temperature observations for multiple days. To do so, we will provide a sequence of reference evapotranspiration rasters, e.g., `ref_et0.tif` and `ref_et1.tif`. For temperature maps, we can calibrate the model to best fit the maps of `T0.tif` and `T1.tif` as in:

.. code-block:: bash

    $ invest-ucm-calibration lulc.tif biophysical-table.csv factors \
                --ref-et-raster-filepaths ref_et0.tif ref_et1.tif \
                --t-raster-filepaths T0.tif T1.tif --num-steps 10 \
                --num-update-logs 10 --dst-filepath calibrated-params.json

Similarly, if we have the station measurements for the two days of `ref_et0.tif` and `ref_et1.tif` (see the file `station-t.csv`), we can calibrate the model to best fit the measurements of the two days as in:

.. code-block:: bash
                
    $ invest-ucm-calibration lulc.tif biophysical-table.csv factors \
                --ref-et-raster-filepaths ref_et0.tif ref_et1.tif \
                --station-t-filepath station-t.csv \
                --station-locations station-locations.csv --num-steps 10 \
                --num-update-logs 10 --dst-filepath calibrated-params.json

----------------------------------------------------------
Providing custom reference temperatures and UHI magnitudes
----------------------------------------------------------

By default, the reference temperature and UHI magnitude (parameters of the urban cooling model) for each day will be automatically extracted from the temperature observations (i.e., the reference temperature will be set as the minimum observed temperature while the UHI magnitude will be set as the difference between the maximum and minimum observed temperatures) both when calibrating with temperature maps or station measurements. Such behavior can be overridden by explicitly providing such values as in:

.. code-block:: bash

    $ invest-ucm-calibration lulc.tif biophysical-table.csv factors \
                --ref-et-raster-filepaths ref_et0.tif ref_et1.tif --t-refs 22 20 \
                --uhi-maxs 5 6 --t-raster-filepaths T0.tif T1.tif --num-steps 10 \
                --num-update-logs 10 --dst-filepath calibrated-params.json
