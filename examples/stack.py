#!/usr/bin/env python

import sys
import argparse
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stack result files')
    parser.add_argument('ofname', type=str, help='output file name')
    parser.add_argument('ifnames', type=str, nargs='+',
                        help='input file names')
    args = parser.parse_args()

    try:
        np.load(args.ofname)
    except FileNotFoundError:
        pass
    else:
        print(f'Output file `{args.ofname}` already exists;'
              ' Delete explicitly.')
        sys.exit(1)

    print('Inputs:')
    inputs = []
    for ifname in args.ifnames:
        try:
            input_ = np.load(ifname)
        except FileNotFoundError:
            print(f' - NOT FOUND - {ifname}')
        else:
            inputs.append(input_)
            print(f' - {input_.shape} - {ifname}')

    print('Output:')
    output = np.vstack(inputs)
    print(f' - {output.shape} - {args.ofname}')

    with open(args.ofname, 'bw') as f:
        np.save(f, output)
