#!/usr/bin/env python

import argparse

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('dark')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+')

    args = parser.parse_args()

    for name in args.results:
        try:
            results = np.load(name)
        except FileNotFoundError:
            print(f'File {name} not found.  Skipping..')
            continue

        print(f'File {name} loaded.')

        mean = results.mean(axis=0)
        plt.plot(mean, label=name)

    plt.grid()
    plt.legend()
    plt.show()
