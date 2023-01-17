from ctypes import CDLL, RTLD_GLOBAL
import sys, os

platform = sys.platform

libname = {'linux':'libmkl_rt.so', # works for python3 on linux
           'linux2':'libmkl_rt.so', # works for python2 on linux
           'darwin':'libmkl_rt.dylib',
           'win32':'mkl_rt.dll'}

def _loadMKL():
    MKLlib = None
    try:
        # Look for MKL in path
        MKLlib = CDLL(libname[platform])
    except:
        try:
            if platform == 'win32' and MKLlib is None:
                try:
                    MKLlib = CDLL('mkl_rt.1.dll')
                except:
                    MKLlib = CDLL('mkl_rt.2.dll')
                
            # Look for anaconda mkl if the MKLlib didn't load at this point
            if 'Anaconda' in sys.version and MKLlib is None:
                if platform in ['linux', 'linux2','darwin']:
                    libpath = ['/']+sys.executable.split('/')[:-2] + \
                              ['lib',libname[platform]]
                elif platform == 'win32':
                    libpath = sys.executable.split(os.sep)[:-1] + \
                              ['Library','bin',libname[platform]]
                MKLlib = CDLL(os.path.join(*libpath))
        except Exception as e: 
            raise e

    if MKLlib is None:
        raise FileNotFoundError(f"Couldn't find the MKL dll.")
    return MKLlib