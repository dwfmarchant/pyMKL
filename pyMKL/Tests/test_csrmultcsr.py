from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import unittest
import pyMKL
import scipy.sparse as sp
import numpy as np

class Test_dcsrmulcsr(unittest.TestCase):

    nSize = 500
    A = sp.rand(nSize, nSize, 0.05, format='csr', dtype=np.float64, random_state=1)
    B = sp.rand(nSize, nSize, 0.05, format='csr', dtype=np.float64, random_state=2)

    a = A.data
    ja = A.indices+1
    ia = A.indptr+1
    b = B.data
    jb = B.indices+1
    ib = B.indptr+1
    m,n,k = 3*[nSize]

    def test_AmulB(self):
        C_sp = self.A*self.B
        c, jc, ic = pyMKL.mkl_dcsrmultcsr(self.m, self.n, self.k, 
                                          self.a, self.ja, self.ia, 
                                          self.b, self.jb, self.ib)
        C_mkl = sp.csr_matrix((c, jc-1,ic-1),(self.m,self.k))
        diff = C_sp - C_mkl
        self.assertEquals(diff.nnz,0)

    def test_ATmulB(self):
        C_sp = self.A.T*self.B
        c, jc, ic = pyMKL.mkl_dcsrmultcsr(self.m, self.n, self.k, 
                                          self.a, self.ja, self.ia, 
                                          self.b, self.jb, self.ib,
                                          trans='T')
        C_mkl = sp.csr_matrix((c, jc-1,ic-1),(self.m,self.k))
        diff = C_sp - C_mkl
        self.assertEquals(diff.nnz,0)

if __name__ == '__main__':
    unittest.main()
