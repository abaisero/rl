#!/usr/bin/env python

import argparse
import itertools as itt

import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style('dark')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=None, help='output file')
    parser.add_argument('--horizon', type=int, help='graph horizon')

    parser.add_argument('--nosharex', action='store_true')
    parser.add_argument('--nosharey', action='store_true')

    parser.add_argument('fnames', metavar='F', nargs='+', help='result files')

    config = parser.parse_args()
    print(config)

    sharex = not config.nosharex
    sharey = not config.nosharey
    axis = 0, 2

    data = {}
    for fname in config.fnames:
        try:
            results = torch.load(fname)
        except FileNotFoundError:
            print(f'File {fname} not found.  Skipping..')
            continue

        key = (results['config'].algo,
               results['config'].lra,
               results['config'].lrb)
        returns = results['returns'].numpy()
        data[key] = dict(
            perc=np.percentile(returns, [0, 25, 50, 75, 100], axis=axis),
            mean=returns.mean(axis),
            min=returns.min(axis),
            max=returns.max(axis),
        )

    algos = sorted(set(key[0] for key in data.keys()))
    lras = sorted(set(key[1] for key in data.keys()))
    lrbs = sorted(set(key[2] for key in data.keys()))
    nas = len(lras)
    nbs = len(lrbs)

    for algo in algos:
        fig, axes = plt.subplots(nas, nbs, sharex=sharex, sharey=sharey)
        fig.suptitle(algo)

        for i, j in itt.product(range(nas), range(nbs)):
            ax = axes[i, j]
            lra, lrb = lras[i], lrbs[j]

            key = algo, lra, lrb
            if key in data:
                ax.plot(data[key]['perc'][2], linewidth=1)
                ax.grid(True)
                ndata = data[key]['mean'].size
                ax.fill_between(range(ndata),
                                data[key]['perc'][0],
                                data[key]['perc'][4],
                                alpha=.2)
                ax.fill_between(range(ndata),
                                data[key]['perc'][1],
                                data[key]['perc'][3],
                                alpha=.2)

            if j == 0:
                ax.set_ylabel(f'lra={lra}')
            if i == nas - 1:
                ax.set_xlabel(f'lrb={lrb}')

    plt.show()
