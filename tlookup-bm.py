import torch
import numpy as np

from fileio_utils import * 

import os, sys

import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('dim', type=int)
parser.add_argument('low', type=int)
parser.add_argument('len', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
    
        NUM, DIM = args.num, args.dim
        LOW, LEN = args.low, args.len

        for i in range(NUM):
            A = torch.randint(low = LOW, high = LOW+LEN, size = (DIM, ))
            save_int(A, 1, os.path.join(tempdir, f'A_{i}.bin'))

        compilation_error = os.system('make tlookup-bm')
        if compilation_error:
            print('Compilation error')
            sys.exit(1)

        prefix = os.path.join(tempdir, f'A_')
        os.system(f'./tlookup-bm {prefix} {NUM} {DIM} {LOW} {LEN}')