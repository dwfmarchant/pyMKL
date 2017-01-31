# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object

from pyMKL import pardisoinit, pardiso
from ctypes import POINTER, byref, c_longlong, c_int
import scipy.sparse as sp
from numpy import ctypeslib

import numpy as np
import pandas as pd
from pyMKL import pardisoSolver

from datetime import datetime



mtypes = {'spd':2,
          'symm':-2,
          'unsymm':11}

class my_pardisoSolver(pardisoSolver):
    def __init__(self, A, my_iparm2, my_iparm10, my_iparm11, my_iparm13,
                 my_iparm8, my_iparm21, my_iparm27,
                 mtype=11, verbose=False, pivoting=False):

        self.mtype = mtype
        if mtype in [1, 3]:
            msg = "mtype = 1/3 - structurally symmetric matrices not supported"
            raise NotImplementedError(msg)
        elif mtype in [2, -2, 4, -4, 6, 11, 13]:
            pass
        else:
            msg = "Invalid mtype: mtype={}".format(mtype)
            raise ValueError(msg)


        self.n = A.shape[0]

        if mtype in [4, -4, 6, 13]:
            # Complex matrix
            self.dtype = np.complex128
        elif mtype in [2, -2, 11]:
            # Real matrix
            self.dtype = np.float64
        self.ctypes_dtype = ctypeslib.ndpointer(self.dtype)

        # If A is symmetric, store only the upper triangular portion
        if mtype in [2, -2, 4, -4, 6]:
            A = sp.triu(A, format='csr')
        elif mtype in [11, 13]:
            A = A.tocsr()

        if not A.has_sorted_indices:
            A.sort_indices()

        self.a = A.data
        self.ia = A.indptr
        self.ja = A.indices

        self._MKL_a = self.a.ctypes.data_as(self.ctypes_dtype)
        self._MKL_ia = self.ia.ctypes.data_as(POINTER(c_int))
        self._MKL_ja = self.ja.ctypes.data_as(POINTER(c_int))

        # Hardcode some parameters for now...
        self.maxfct = 1
        self.mnum = 1
        self.perm = 0

        if verbose:
            self.msglvl = 1
        else:
            self.msglvl = 0

        # Initialize handle to data structure
        self.pt = np.zeros(64, np.int64)
        self._MKL_pt = self.pt.ctypes.data_as(POINTER(c_longlong))

        # Initialize parameters
        self.iparm = np.zeros(64, dtype=np.int32)
        self._MKL_iparm = self.iparm.ctypes.data_as(POINTER(c_int))

        # Initialize pardiso
        pardisoinit(self._MKL_pt, byref(c_int(self.mtype)), self._MKL_iparm)

        # Set iparm
        self.iparm[1] = 3 # Use parallel nested dissection for reordering
        self.iparm[23] = 1 # Use parallel factorization
        self.iparm[34] = 1 # Zero base indexing


        # For constrained systems with highly indefinite symmetric matrices
        if pivoting:
            self.iparm[1]  = my_iparm2

            self.iparm[9]  = my_iparm10


            self.iparm[10] = my_iparm11
            self.iparm[12] = my_iparm13


            self.iparm[7]  = my_iparm8
            self.iparm[20] = my_iparm21

            self.iparm[26] = my_iparm27


def solve_sparse(A, b, my_iparm2, my_iparm10, my_iparm11, my_iparm13,
                 my_iparm8, my_iparm21, my_iparm27,
                 matrix_type='symm', verbose=False, pivoting=False):

    mtype = mtypes[matrix_type]

    pSolve = my_pardisoSolver(A, my_iparm2, my_iparm10, my_iparm11, my_iparm13,
                 my_iparm8, my_iparm21, my_iparm27,
                 mtype=mtype, verbose=verbose, pivoting=pivoting)

    x = pSolve.run_pardiso(13, b)
    pSolve.clear()

    return x


def save_sparse_csr(filename,array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape']), loader['res']

my_favorite_S_and_res = 'my_favorite_S_and_res.npz'

S, res = load_sparse_csr(my_favorite_S_and_res)



my_iparm2_list  = np.array([0, 1, 2, 3], dtype='int')
my_iparm8_list  = np.array([0, 10], dtype='int')
my_iparm10_list = np.array([8, 13], dtype='int')
my_iparm11_list = np.array([0, 1], dtype='int')
my_iparm13_list = np.array([0, 1], dtype='int')
my_iparm21_list = np.array([0, 1, 2, 3], dtype='int')
my_iparm27_list = np.array([0], dtype='int')

iparm2_sym  = []
iparm8_sym  = []
iparm10_sym = []
iparm11_sym = []
iparm13_sym = []
iparm21_sym = []
iparm27_sym = []
r_sym       = []
time_taken  = []

for my_iparm2 in my_iparm2_list:
    for my_iparm8 in my_iparm8_list:
        for my_iparm10 in my_iparm10_list:
            for my_iparm11 in my_iparm11_list:
                for my_iparm13 in my_iparm13_list:
                    for my_iparm21 in my_iparm21_list:
                        for my_iparm27 in my_iparm27_list:
                            
                            startTime = datetime.now()
                            
                            delta_q = solve_sparse(S, res, my_iparm2, my_iparm10,
                                                   my_iparm11, my_iparm13,
                                                   my_iparm8, my_iparm21,
                                                   my_iparm27, pivoting=True)
                            
                            time_taken.append(datetime.now() - startTime)

                            r = np.linalg.norm(S @ delta_q - res)

                            iparm2_sym.append(my_iparm2)
                            iparm8_sym.append(my_iparm8)
                            iparm10_sym.append(my_iparm10)
                            iparm11_sym.append(my_iparm11)
                            iparm13_sym.append(my_iparm13)
                            iparm21_sym.append(my_iparm21)
                            iparm27_sym.append(my_iparm27)
                            r_sym.append(r)

                            

startTime = datetime.now()

delta_q = solve_sparse(S, res, my_iparm2, my_iparm10,
                       my_iparm11, my_iparm13,
                       my_iparm8, my_iparm21,
                       my_iparm27, pivoting=False, matrix_type='unsymm')

time_taken.append(datetime.now() - startTime)

r = np.linalg.norm(S @ delta_q - res)

iparm2_sym.append(my_iparm2)
iparm8_sym.append(my_iparm8)
iparm10_sym.append(my_iparm10)
iparm11_sym.append(my_iparm11)
iparm13_sym.append(my_iparm13)
iparm21_sym.append(my_iparm21)
iparm27_sym.append(my_iparm27)
r_sym.append(r)

my_data = {'residuum': r_sym,
           'iparm2'  : iparm2_sym,
           'iparm8'  : iparm8_sym,
           'iparm10' : iparm10_sym,
           'iparm11' : iparm11_sym,
           'iparm13' : iparm13_sym,
           'iparm21' : iparm21_sym,
           'iparm27' : iparm27_sym,
           'time' : time_taken,
           }
           
my_dataframe = pd.DataFrame(my_data)

print(my_dataframe.sort_values('residuum', ascending=True)[:30])

