#!/usr/bin/env python
# encoding: utf-8

from setuptools import setup


setup(
    name='rl',
    version='0.0.1',
    description='rl - reinforcement learning code and programming exercises '
                'from Sutton and Barto',
    # long_description=read('README.rst'),
    author='Andrea Baisero',
    author_email='andrea.baisero@gmail.com',
    url='https://github.com/bigblindbais/rl',

    packages=['rl'],
    package_dir={'': 'src'},

    package_data={'rl': [
        'data/pomdp/*.pomdp',
        'data/mdp/*.mpd',
        'data/fss/*.fss',
        'data/fsc/*.fsc',
    ]},
    include_package_data=True,

    scripts=[
        'scripts/rl-pomdp',
        'scripts/rl-fsc',
        'scripts/rl-fss',
        'scripts/evaluate.py',
        'scripts/pgradient.py',
        'scripts/shapes.py',
        'scripts/results.py',
        'scripts/discounted.py',
        'scripts/discounted_memreplay.py',
    ],

    install_requires=['numpy', 'scipy', 'pyqt5', 'pyqtgraph'],  # pytk
    license='MIT',
    test_suite='tests',
)
