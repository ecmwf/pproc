from setuptools import setup, find_packages
import os

setup(
    name="pproc",
    version='1.1.0',
    author='ECMWF',
    description="ECMWF Post-processing tools",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        "filelock==3.12.0",
        "code-meters",
        "earthkit-meteo>=0.1.0"
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
            "pproc-quantiles=pproc.quantiles:main",
            "pproc-tcycl-run=pproc.tcycl.tcycl_run:main",
            "pproc-tcycl-summarise-ibtks=pproc.tcycl.tcycl_summarise_ibtks:main",
            "pproc-tcycl-summarise-tcycl=pproc.tcycl.tcycl_summarise_tcycl:main",
            "pproc-tcycl-summarise-trckr=pproc.tcycl.tcycl_summarise_trckr:main",
            "pproc-tcycl-evaluate=pproc.tcycl.tcycl_evaluate:main",
            "pproc-thermal-indices=pproc.thermal_indices:main",
        ],
    },
)
