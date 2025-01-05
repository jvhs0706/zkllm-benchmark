import torch
import numpy as np

from fileio_utils import * 

import os, sys

import argparse
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('num', type=int)
parser.add_argument('dim', type=int)
parser.add_argument('gen_dim', type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tempdir:
        
        NUM, DIM = args.num, args.dim
        GEN_DIM = args.gen_dim

        for i in range(NUM):
            A = torch.randn(DIM).cuda()
            save_int(A, 65536, os.path.join(tempdir, f'A_{i}.bin'))

        compilation_error = os.system('make com-bm')
        if compilation_error:
            print('Compilation error')
            sys.exit(1)

        prefix = os.path.join(tempdir, f'A_')
        prefix_com = os.path.join(tempdir, f'com_')
        os.system(f'./com-bm {prefix} {prefix_com} {NUM} {DIM} {GEN_DIM}')