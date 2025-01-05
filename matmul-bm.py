import torch
import numpy as np

from fileio_utils import * 

import os, sys

import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('dim0', type=int)
parser.add_argument('dim1', type=int)
parser.add_argument('dim2', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
    
        NUM = args.num
        DIM0, DIM1, DIM2 = args.dim0, args.dim1, args.dim2

        for i in range(NUM):
            A = torch.randn(DIM0, DIM1).cuda()
            B = torch.randn(DIM1, DIM2).cuda()
            save_int(A, 65536, os.path.join(tempdir, f'A_{i}.bin'))
            save_int(B, 65536, os.path.join(tempdir, f'B_{i}.bin'))

        compilation_error = os.system('make matmul-bm')
        if compilation_error:
            print('Compilation error')
            sys.exit(1)

        prefix_A, prefix_B = os.path.join(tempdir, f'A_'), os.path.join(tempdir, f'B_')
        os.system(f'./matmul-bm {prefix_A} {prefix_B} {NUM} {DIM0} {DIM1} {DIM2}')