from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from . import MKLlib
from ctypes import Structure, POINTER, byref, c_int, c_char, c_double, c_float
import numpy as np

_mkl_dcsrmultcsr = MKLlib.mkl_dcsrmultcsr
_mkl_dcsrmultcsr.argtypes = [POINTER(c_char),   #trans
                             POINTER(c_int),     #request
                             POINTER(c_int),     #sort
                             POINTER(c_int),     #m
                             POINTER(c_int),     #n
                             POINTER(c_int),     #k
                             POINTER(c_double),  #a
                             POINTER(c_int),     #ja
                             POINTER(c_int),     #ia
                             POINTER(c_double),  #b
                             POINTER(c_int),     #jb
                             POINTER(c_int),     #ib
                             POINTER(c_double),  #c
                             POINTER(c_int),     #jc
                             POINTER(c_int),     #ic
                             POINTER(c_int),     #nzmax
                             POINTER(c_int)]     #info 
_mkl_dcsrmultcsr.restype = None

def mkl_dcsrmultcsr(m, n, k, a, ja, ia, b, jb, ib, trans=b'N', sort=7):

    trans_point = byref(c_char(bytes(trans)))
    sort_point = byref(c_int(sort))
    m_point = byref(c_int(m))
    n_point = byref(c_int(n))
    k_point = byref(c_int(k))

    a_point = a.ctypes.data_as(POINTER(c_double))
    ja_point = ja.ctypes.data_as(POINTER(c_int))
    ia_point = ia.ctypes.data_as(POINTER(c_int))

    b_point = b.ctypes.data_as(POINTER(c_double))
    jb_point = jb.ctypes.data_as(POINTER(c_int))
    ib_point = ib.ctypes.data_as(POINTER(c_int))
    
    # First run to determine nnz(C)
    request = 1
    request_point = byref(c_int(request))

    c = np.zeros(1, dtype=c_double)
    jc = np.zeros(1, dtype=c_int)
    ic = np.zeros(m+1, dtype=c_int)
    nzmax = 0

    c_point = c.ctypes.data_as(POINTER(c_double))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    ic_point = ic.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_dcsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point) 

    nnz_c = ic[-1]-1

    # Second run to calculate product
    request = 2
    request_point = byref(c_int(request))

    c = np.zeros(nnz_c, dtype=c_double)
    jc = np.zeros(nnz_c, dtype=c_int)
    nzmax = nnz_c

    c_point = c.ctypes.data_as(POINTER(c_double))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_dcsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point)

    return c, jc, ic


_mkl_scsrmultcsr = MKLlib.mkl_scsrmultcsr
_mkl_scsrmultcsr.argtypes = [POINTER(c_char),   #trans
                             POINTER(c_int),     #request
                             POINTER(c_int),     #sort
                             POINTER(c_int),     #m
                             POINTER(c_int),     #n
                             POINTER(c_int),     #k
                             POINTER(c_float),  #a
                             POINTER(c_int),     #ja
                             POINTER(c_int),     #ia
                             POINTER(c_float),  #b
                             POINTER(c_int),     #jb
                             POINTER(c_int),     #ib
                             POINTER(c_float),  #c
                             POINTER(c_int),     #jc
                             POINTER(c_int),     #ic
                             POINTER(c_int),     #nzmax
                             POINTER(c_int)]     #info 
_mkl_scsrmultcsr.restype = None

def mkl_scsrmultcsr(m, n, k, a, ja, ia, b, jb, ib, trans=b'N', sort=7):

    trans_point = byref(c_char(bytes(trans)))
    sort_point = byref(c_int(sort))
    m_point = byref(c_int(m))
    n_point = byref(c_int(n))
    k_point = byref(c_int(k))

    a_point = a.ctypes.data_as(POINTER(c_float))
    ja_point = ja.ctypes.data_as(POINTER(c_int))
    ia_point = ia.ctypes.data_as(POINTER(c_int))

    b_point = b.ctypes.data_as(POINTER(c_float))
    jb_point = jb.ctypes.data_as(POINTER(c_int))
    ib_point = ib.ctypes.data_as(POINTER(c_int))
    
    # First run to determine nnz(C)
    request = 1
    request_point = byref(c_int(request))

    c = np.zeros(1, dtype=c_float)
    jc = np.zeros(1, dtype=c_int)
    ic = np.zeros(m+1, dtype=c_int)
    nzmax = 0

    c_point = c.ctypes.data_as(POINTER(c_float))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    ic_point = ic.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_scsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point) 

    nnz_c = ic[-1]-1

    # Second run to calculate product
    request = 2
    request_point = byref(c_int(request))

    c = np.zeros(nnz_c, dtype=c_float)
    jc = np.zeros(nnz_c, dtype=c_int)
    nzmax = nnz_c

    c_point = c.ctypes.data_as(POINTER(c_float))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_scsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point)

    return c, jc, ic


class MKL_Complex16(Structure):
    _fields_ = [('real',c_double),
                ('imag',c_double)]

_mkl_zcsrmultcsr = MKLlib.mkl_zcsrmultcsr
_mkl_zcsrmultcsr.argtypes = [POINTER(c_char),   #trans
                             POINTER(c_int),     #request
                             POINTER(c_int),     #sort
                             POINTER(c_int),     #m
                             POINTER(c_int),     #n
                             POINTER(c_int),     #k
                             POINTER(MKL_Complex16),  #a
                             POINTER(c_int),     #ja
                             POINTER(c_int),     #ia
                             POINTER(MKL_Complex16),  #b
                             POINTER(c_int),     #jb
                             POINTER(c_int),     #ib
                             POINTER(MKL_Complex16),  #c
                             POINTER(c_int),     #jc
                             POINTER(c_int),     #ic
                             POINTER(c_int),     #nzmax
                             POINTER(c_int)]     #info 
_mkl_zcsrmultcsr.restype = None

def mkl_zcsrmultcsr(m, n, k, a, ja, ia, b, jb, ib, trans='N', sort=7):

    trans_point = byref(c_char(bytes(trans)))
    sort_point = byref(c_int(sort))
    m_point = byref(c_int(m))
    n_point = byref(c_int(n))
    k_point = byref(c_int(k))

    a_point = a.ctypes.data_as(POINTER(MKL_Complex16))
    ja_point = ja.ctypes.data_as(POINTER(c_int))
    ia_point = ia.ctypes.data_as(POINTER(c_int))

    b_point = b.ctypes.data_as(POINTER(MKL_Complex16))
    jb_point = jb.ctypes.data_as(POINTER(c_int))
    ib_point = ib.ctypes.data_as(POINTER(c_int))
    
    # First run to determine nnz(C)
    request = 1
    request_point = byref(c_int(request))

    c = np.zeros(1, dtype=np.complex128)
    jc = np.zeros(1, dtype=c_int)
    ic = np.zeros(m+1, dtype=c_int)
    nzmax = 0

    c_point = c.ctypes.data_as(POINTER(MKL_Complex16))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    ic_point = ic.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_zcsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point) 

    nnz_c = ic[-1]-1

    # Second run to calculate product
    request = 2
    request_point = byref(c_int(request))

    c = np.zeros(nnz_c, dtype=np.complex128)
    jc = np.zeros(nnz_c, dtype=c_int)
    nzmax = nnz_c

    c_point = c.ctypes.data_as(POINTER(MKL_Complex16))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_zcsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point)

    return c, jc, ic


class MKL_Complex8(Structure):
    _fields_ = [('real',c_float),
                ('imag',c_float)]

_mkl_ccsrmultcsr = MKLlib.mkl_ccsrmultcsr
_mkl_ccsrmultcsr.argtypes = [POINTER(c_char),   #trans
                             POINTER(c_int),     #request
                             POINTER(c_int),     #sort
                             POINTER(c_int),     #m
                             POINTER(c_int),     #n
                             POINTER(c_int),     #k
                             POINTER(MKL_Complex8),  #a
                             POINTER(c_int),     #ja
                             POINTER(c_int),     #ia
                             POINTER(MKL_Complex8),  #b
                             POINTER(c_int),     #jb
                             POINTER(c_int),     #ib
                             POINTER(MKL_Complex8),  #c
                             POINTER(c_int),     #jc
                             POINTER(c_int),     #ic
                             POINTER(c_int),     #nzmax
                             POINTER(c_int)]     #info 
_mkl_ccsrmultcsr.restype = None

def mkl_ccsrmultcsr(m, n, k, a, ja, ia, b, jb, ib, trans='N', sort=7):

    trans_point = byref(c_char(bytes(trans)))
    sort_point = byref(c_int(sort))
    m_point = byref(c_int(m))
    n_point = byref(c_int(n))
    k_point = byref(c_int(k))

    a_point = a.ctypes.data_as(POINTER(MKL_Complex8))
    ja_point = ja.ctypes.data_as(POINTER(c_int))
    ia_point = ia.ctypes.data_as(POINTER(c_int))

    b_point = b.ctypes.data_as(POINTER(MKL_Complex8))
    jb_point = jb.ctypes.data_as(POINTER(c_int))
    ib_point = ib.ctypes.data_as(POINTER(c_int))
    
    # First run to determine nnz(C)
    request = 1
    request_point = byref(c_int(request))

    c = np.zeros(1, dtype=np.complex64)
    jc = np.zeros(1, dtype=c_int)
    ic = np.zeros(m+1, dtype=c_int)
    nzmax = 0

    c_point = c.ctypes.data_as(POINTER(MKL_Complex8))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    ic_point = ic.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_ccsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point) 

    nnz_c = ic[-1]-1

    # Second run to calculate product
    request = 2
    request_point = byref(c_int(request))

    c = np.zeros(nnz_c, dtype=np.complex64)
    jc = np.zeros(nnz_c, dtype=c_int)
    nzmax = nnz_c

    c_point = c.ctypes.data_as(POINTER(MKL_Complex8))
    jc_point = jc.ctypes.data_as(POINTER(c_int))
    nzmax_point = byref(c_int(nzmax))

    info  = c_int(-3)
    info_point  = byref(info)

    _mkl_ccsrmultcsr(trans_point, request_point, sort_point, m_point, 
                     n_point, k_point, a_point, ja_point, ia_point, 
                     b_point, jb_point, ib_point, c_point, jc_point, 
                     ic_point, nzmax_point, info_point)

    return c, jc, ic




