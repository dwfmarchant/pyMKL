from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

from ctypes import Structure, POINTER, c_int, c_char_p
from . import MKLlib

class pyMKLVersion(Structure):
    _fields_ = [('MajorVersion',c_int),
                ('MinorVersion',c_int),
                ('UpdateVersion',c_int),
                ('ProductStatus',c_char_p),
                ('Build',c_char_p),
                ('Processor',c_char_p),
                ('Platform',c_char_p)]
_mkl_get_version = MKLlib.mkl_get_version
_mkl_get_version.argtypes = [POINTER(pyMKLVersion)]
_mkl_get_version.restype = None

def mkl_get_version():
    MKLVersion = pyMKLVersion()
    _mkl_get_version(MKLVersion)
    version = {'MajorVersion':MKLVersion.MajorVersion,
               'MinorVersion':MKLVersion.MinorVersion,
               'UpdateVersion':MKLVersion.UpdateVersion,
               'ProductStatus':MKLVersion.ProductStatus,
               'Build':MKLVersion.Build,
               'Platform':MKLVersion.Platform}

    versionString = 'Intel(R) Math Kernel Library Version {MajorVersion}.{MinorVersion}.{UpdateVersion} {ProductStatus} Build {Build} for {Platform} applications'.format(**version)

    return versionString
