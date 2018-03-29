#!/usr/bin/env python

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('dark')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--H', type=int)
    parser.add_argument('results', nargs='+')

    args = parser.parse_args()

    for fname in args.results:
        try:
            results = np.load(fname)
        except FileNotFoundError:
            print(f'File {fname} not found.  Skipping..')
            continue

        bname = os.path.basename(fname)
        print(f'Array `{bname}` loaded:')
        print(f' - shape: {results.shape}')

        if args.H is not None:
            results = results[:, :args.H]

        plt.subplot(311)
        res = results.max(axis=0)
        plt.plot(res, label=bname)
        plt.grid(True)

        plt.subplot(312)
        res = results.mean(axis=0)
        plt.plot(res, label=bname)
        plt.grid(True)

        plt.subplot(313)
        res = results.min(axis=0)
        plt.plot(res, label=bname)
        plt.grid(True)

    plt.legend(frameon=True)
    plt.show()
