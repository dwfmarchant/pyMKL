from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()

import unittest
import pyMKL

class TestMKLUtils(unittest.TestCase):

    def test_mkl_get_version(self):
        pyMKL.mkl_get_version()

if __name__ == '__main__':
    unittest.main()
