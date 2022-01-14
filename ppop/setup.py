from setuptools import setup, find_packages
import os

setup(
    name="pproc",
    version='0.1',
    author='ECMWF',
    description="ECMWF Post-processing tools",
    packages=find_packages(exclude=["test_*", "*.tests", "*.tests.*", "tests.*", "tests"]),
    scripts=[os.path.join('bin', i) for i in os.listdir('bin')],
    install_requires=[
    ],
    tests_require=[
    ],
)
