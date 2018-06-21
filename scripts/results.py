#!/usr/bin/env python

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import seaborn as sns
sns.set_style('dark')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=None, help='output file')
    parser.add_argument('--horizon', type=int, help='graph horizon')
    parser.add_argument('--mm', action='store_true', help='show min and max')
    parser.add_argument('--line', metavar=('C', 'F'), nargs=2, action='append',
                        dest='lines', default=[], help='graph reference lines')

    parser.add_argument('fnames', metavar='F', nargs='+', help='result files')

    config = parser.parse_args()
    print(config)
    axes = []

    for fname in config.fnames:
        try:
            results = torch.load(fname).numpy()
        except FileNotFoundError:
            print(f'File {fname} not found.  Skipping..')
            continue

        axis = 0, 2

        bname = os.path.basename(fname)
        print(f'Array `{bname}` loaded:')
        print(f' - shape: {results.shape}')

        if config.horizon is not None:
            results = results[:, :config.horizon]

        if config.mm:
            plt.subplot(311)
            axes.append(plt.gca())
            res = results.max(axis=axis)
            plt.plot(res, label=bname, linewidth=1)
            plt.grid(True)

            plt.subplot(312)

        axes.append(plt.gca())
        res = results.mean(axis=axis)
        plt.plot(res, label=bname, linewidth=1)
        plt.grid(True)

        if config.mm:
            plt.subplot(313)
            axes.append(plt.gca())
            res = results.min(axis=axis)
            plt.plot(res, label=bname, linewidth=1)
            plt.grid(True)

    for color, value in config.lines:
        try:
            data = np.load(value)
            import ipdb
            ipdb.set_trace()
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

    if config.out:
        plt.savefig(config.out, bbox_inches='tight')

    plt.show()
