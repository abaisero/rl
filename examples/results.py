#!/usr/bin/env python

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('dark')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', type=int)
    parser.add_argument('--mm', action='store_true')
    parser.add_argument('--line', nargs=2, action='append', dest='lines',
                        default=[], help='plot lines')

    parser.add_argument('results', nargs='+')

    args = parser.parse_args()
    axes = []

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

        if args.mm:
            plt.subplot(311)
            axes.append(plt.gca())
            res = results.max(axis=0)
            plt.plot(res, label=bname, linewidth=1)
            plt.grid(True)

            plt.subplot(312)

        axes.append(plt.gca())
        res = results.mean(axis=0)
        plt.plot(res, label=bname, linewidth=1)
        plt.grid(True)

        if args.mm:
            plt.subplot(313)
            axes.append(plt.gca())
            res = results.min(axis=0)
            plt.plot(res, label=bname, linewidth=1)
            plt.grid(True)

    for color, value in args.lines:
        try:
            data = np.load(value)
        except FileNotFoundError:
            for ax in axes:
                ax.axhline(value, color=color, linewidth=1)
        else:
            datap = np.percentile(data, [0, 25, 50, 75, 100])
            for ax in axes:
                ax.axhline(datap[0], color=color, linewidth=1, linestyle=':')
                ax.axhline(datap[1], color=color, linewidth=1, linestyle=':')
                ax.axhline(datap[2], color=color, linewidth=1)
                ax.axhline(datap[3], color=color, linewidth=1, linestyle=':')
                ax.axhline(datap[4], color=color, linewidth=1, linestyle=':')

    plt.legend(frameon=True)
    plt.show()
