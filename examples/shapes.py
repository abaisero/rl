#!/usr/bin/env python

import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prints .npy shapes')
    parser.add_argument('fnames', type=str, nargs='+', help='file names')
    args = parser.parse_args()

    print('Files:')
    inputs = []
    for fname in args.fnames:
        try:
            a = np.load(fname)
        except FileNotFoundError:
            print(f' - NOT FOUND - {fname}')
        except OSError:
            print(f' - NOT ARRAY - {fname}')
        else:
            print(f' - {a.shape} - {fname}')
