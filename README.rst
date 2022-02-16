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
  python3 -m pip install cython

..
  FIXME 
  python3 -m venv --system-site-packages venv

Install requirements
--------------------

Install the pproc package requirements::

  python3 -m pip install -r requirements.txt

..
  FIXME this should replace the "Install" sections, incl. cython?
  Needs populating requirements.txt

Install pyfdb
-------------

Clone and install pyfdb::

  git clone ssh://git@git.ecmwf.int/mars/pyfdb.git
  cd pyfdb
  git checkout develop
  python3 -m pip install -e .
  cd ..

Install mir-python
------------------

Clone, build and install mir-python::

  git clone ssh://git@git.ecmwf.int/mir/mir-python.git
  cd mir-python
  export MIR_INCLUDE_DIRS=/path/to/pproc-bundle/install/include
  export MIR_LIB_DIR=/path/to/pproc-bundle/install/lib64
  python setup.py install
  cd ..

Examples
========

fdb
---

This example assumes that an FDB is set up and contains a field (see the
request in the example script). The field is read using pyfdb, interpolated to
a regular grid, then read into eccodeshl::

  export LD_LIBRARY_PATH=/path/to/pproc-bundle/install/lib64:$LD_LIBRARY_PATH
  python3 test_fdb_mir.py


pts
---

This example creates a pts GRIB product, some options are available::

  pproc-pts --help
  pproc-pts examples/pts/msl_05L_ELSA_2021070300 out.grib --distance=2.0e5

