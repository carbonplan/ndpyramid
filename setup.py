#!/usr/bin/env python

"""The setup script"""


import pathlib

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    INSTALL_REQUIRES = f.read().strip().split('\n')

LONG_DESCRIPTION = pathlib.Path('README.md').read_text()
PYTHON_REQUIRES = '>=3.9'

description = 'A small utility for generating ND array pyramids using Xarray and Zarr.'

setup(
    name='ndpyramid',
    description=description,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    maintainer='CarbonPlan',
    maintainer_email='hello@carbonplan.org',
    url='https://github.com/carbonplan/ndpyramid',
    packages=find_packages(),
    include_package_data=True,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    tests_require=['pytest'],
    license='MIT',
    keywords='zarr, xarray, pyramid',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
)
