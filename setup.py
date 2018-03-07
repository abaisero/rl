#!/usr/bin/env python
# encoding: utf-8

import os
from setuptools import setup, find_packages


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='rl',
    version='0.0.1',
    description='rl - reinforcement learning code and programming exercises from Sutton and Barto',
    # long_description=read('README.rst'),
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/bigblindbais/rl',

    packages=['rl'],
    package_dir={'':'src'},

    package_data={'rl': ['data/pomdps/*.pomdp']},
    include_package_data=True,

    scripts=['scripts/rl-pomdps', 'scripts/rl-fscs'],

    install_requires=['numpy', 'scipy'],  #  GPy, pytk
    license='MIT',
    test_suite='tests',
)
