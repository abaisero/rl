#!/usr/bin/env python
# encoding: utf-8

import os
from setuptools import setup


def read(fname):
    """Utility function to read the README file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='rl',
    description='rl - reinforcement learning code and programming exercises from Sutton and Barto',
    # long_description=read('README.rst'),
    author='Andrea Baisero',
    url='https://github.com/bigblindbais/rl',
    download_url='https://github.com/bigblindbais/rl',
    author_email='andrea.baisero@gmail.com',
    version='0.0.1',
    packages=['rl'],
    license='MIT',
    # test_suite='tests',
)
