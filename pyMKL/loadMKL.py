from ctypes import CDLL, RTLD_GLOBAL
import sys, os

platform = sys.platform

libname = {'linux':'libmkl_rt.so', # works for python3 on linux
           'linux2':'libmkl_rt.so', # works for python2 on linux
           'darwin':'libmkl_rt.dylib',
           'win32':'mkl_rt.dll'}

def _loadMKL():

    try:
        # Look for MKL in path
        MKLlib = CDLL(libname[platform])
    except:
        try:
            # Look for anaconda mkl
            cdll_path = os.environ['CONDA_PREFIX']
            if platform in ['linux', 'linux2','darwin']:
                libpath = ['lib']
            elif platform == 'win32':
                libpath = ['Library','bin']
            cdll_path = os.path.join(cdll_path, *libpath, _libname[platform])
            MKLlib = CDLL(cdll_path)
        except Exception as e:
            raise e

    return MKLlib
