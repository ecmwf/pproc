from setuptools import setup, find_packages
import os

setup(
    name="pproc",
    version='0.2.1',
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
            "pproc-extreme=pproc.extreme:main",
            "pproc-interpol=pproc.interpol:main",
            "pproc-pts=pproc.pts:main",
            "pproc-probabilities=pproc.probabilities:main",
            "pproc-ensms=pproc.ensms:main",
            "pproc-wind=pproc.wind:main",
            "pproc-spectrum=pproc.spectrum:main",
            "pproc-clustereps=pproc.clustereps.__main__:main",
            "pproc-clustereps-pca=pproc.clustereps.pca:main",
            "pproc-clustereps-cluster=pproc.clustereps.cluster:main",
            "pproc-clustereps-attr=pproc.clustereps.attribution:main",
            "pproc-anomaly-probabilities=pproc.anomaly_probs:main",
            "pproc-quantiles=pproc.quantiles:main"
        ],
    },
)
