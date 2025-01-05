import torch
import numpy as np

from fileio_utils import * 

import os, sys

import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('dim', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
    
        NUM, DIM = args.num, args.dim

        for i in range(NUM):
            A = torch.randn(DIM).cuda()
            B = torch.randn(DIM).cuda()
            save_int(A, 65536, os.path.join(tempdir, f'A_{i}.bin'))
            save_int(B, 65536, os.path.join(tempdir, f'B_{i}.bin'))

        compilation_error = os.system('make hadamard-bm')
        if compilation_error:
            print('Compilation error')
            sys.exit(1)

        prefix_A, prefix_B = os.path.join(tempdir, f'A_'), os.path.join(tempdir, f'B_')
        os.system(f'./hadamard-bm {prefix_A} {prefix_B} {NUM} {DIM}')