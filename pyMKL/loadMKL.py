from ctypes import CDLL
import sys, os, psutil
import mk

def _loadMKL():
    
    try:
        # from https://github.com/haasad/PyPardisoProject/commit/98701c21ae5d5b9879c51531ace93c2213c89d55
        if sys.platform == 'darwin':
            self.libmkl = CDLL('libmkl_rt.dylib')
        else:
            # find the correct mkl_rt library by searching the loaded libraries of the process
            proc = psutil.Process(os.getpid())
            mkl_rt = [lib.path for lib in proc.memory_maps() if 'mkl_rt' in lib.path][0]
            self.libmkl = CDLL(mkl_rt)
    except Exception as e: 
        raise e

    return MKLlib
