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

        plt.subplot(311)
        res = results.max(axis=0)
        plt.plot(res, label=name)
        plt.grid(True)

        plt.subplot(312)
        res = results.mean(axis=0)
        plt.plot(res, label=name)
        plt.grid(True)

        plt.subplot(313)
        res = results.min(axis=0)
        plt.plot(res, label=name)
        plt.grid(True)

    plt.legend(frameon=True)
    plt.show()
