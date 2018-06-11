#!/usr/bin/env python

import os
import argparse

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fnames', metavar='F', nargs='+', help='result files')

    config = parser.parse_args()

    for fname in config.fnames:
        try:
            rfile = np.load(fname)
        except FileNotFoundError:
            print(f'File {fname} not found.  Skipping..')
            continue

        bname = os.path.basename(fname)
        print(f'Array `{bname}` loaded:')
        try:
            for name, a in rfile.iteritems():
                print(f' - {name}: {a.shape}')
        except AttributeError:
            print(f' - shape: {rfile.shape}`')
