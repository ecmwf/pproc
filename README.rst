=======================
Post-processing toolkit
=======================

Requirements
============

* CMake (3.12 or higher)
* Python 3 (3.7 or higher)

Build instructions
==================

Install pproc-bundle
--------------------

Clone the bundle https://git.ecmwf.int/projects/ECSDK/repos/pproc-bundle/browse. On the ECMWF HPC, load the following 
modules::

  module load intel/2021.4.0 hpcx-openmpi/2.9.0 python3/3.10.10-01 fftw/3.3.9 aec/1.0.6 openblas/0.3.13 tflite/2.13.0

Build and install the underlying C/C++ software stack::

  cd pproc-bundle
  ./pproc-bundle create
  ./pproc-bundle build
  build/install.sh --fast
  cd ..

Create a virtual environment
----------------------------

Create an environment for the Python packages::

  python3 -m venv venv
  source venv/bin/activate

..
  FIXME 
  python3 -m venv --system-site-packages venv

Install requirements
--------------------

Install the pproc package requirements::

  python3 -m pip install -r requirements.txt

Install mir-python
------------------

Build and install mir-python::

  python3 -m pip install cython
  export MIR_INCLUDE_DIRS=/path/to/pproc-bundle/install/include
  export MIR_LIB_DIR=/path/to/pproc-bundle/install/lib64
  python3 -m pip install git+ssh://git@git.ecmwf.int/mir/mir-python.git

Install Python modules
----------------------

Install pproc::

  python3 -m pip install ./pproc

Tests
-----

Run the tests to check everything is working::

  export LD_LIBRARY_PATH=/path/to/pproc-bundle/install/lib64:$LD_LIBRARY_PATH
  python3 -m pytest tests

Examples
========

To use pproc you will need to add the libraries in the pproc-bundle to your `LD_LIBRARY_PATH`::

  export LD_LIBRARY_PATH=/path/to/pproc-bundle/install/lib64:$LD_LIBRARY_PATH

pts
---

This example creates a pts GRIB product, some options are available::

  pproc-pts --help
  pproc-pts out1.grib examples/pts/msl_05L_ELSA_2021070300 --input points --distance=2.0e5
  pproc-pts out2.grib 2022062800/pf/*/* --input tc-tracks --filter-wind 62.2
  pproc-pts out3.grib 2022062800/cf/* --filter-number 1 --filter-basetime "2022-06-28 00" --filter-time 24 72

