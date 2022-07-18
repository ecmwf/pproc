from setuptools import setup, find_packages
import os

setup(
    name="pproc",
    version='0.1',
    author='ECMWF',
    description="ECMWF Post-processing tools",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
    ],
    tests_require=[
    ],
    entry_points={
        "console_scripts": [
            "pproc-extreme-forecast-simple=pproc.extreme_forecast_simple:main",
            "pproc-extreme-forecast=pproc.extreme_forecast:main",
            "pproc-interpol=pproc.interpol:main",
            "pproc-pts=pproc.pts:main",
            "pproc-prob=pproc.prob:main",
            "pproc-spectrum=pproc.spectrum:main",
        ],
    },
)
