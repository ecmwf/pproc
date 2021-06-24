=======================
Post-processing toolkit
=======================

Requirements
============

* CMake (3.12 or higher)
* Python 3

Build instructions
==================

Install pproc-bundle
--------------------

Build and install the underlying C/C++ software stack::

  cd pproc-bundle
  ./bundle-create
  ./bundle-build
  build/install.sh --fast
  cd ..


Create a virtual environment
----------------------------

Create an environment for the Python packages::

  python3 -m venv venv
  source venv/bin/activate
  pip install cython

Install pyfdb
-------------

Clone and install pyfdb::

  git clone ssh://git@git.ecmwf.int/mars/pyfdb.git
  cd pyfdb
  git checkout develop
  pip install -e .
  cd ..

Install pymir
-------------

Clone, build and install pymir::

  git clone ssh://git@git.ecmwf.int/mir/pymir.git
  cd pymir
  export MIR_INCLUDE_DIRS=/path/to/pproc-bundle/install/include
  export MIR_LIB_DIR=/path/to/pproc-bundle/install/lib64
  python setup.py install
  cd ..

Install pyeccodes
-----------------

Needed only for the example::

  pip install pyeccodes

Install eccodes-python
-----------------

Needed only for the example::

  pip install eccodes

Run the example
===============

This example assumes that an FDB is set up and contains a field (see the
request in the example script). The field is read using pyfdb, interpolated to
a regular grid, then read into pyeccodes::

  export LD_LIBRARY_PATH=/path/to/pproc-bundle/install/lib64:$LD_LIBRARY_PATH
  python test_fdb_mir.py

