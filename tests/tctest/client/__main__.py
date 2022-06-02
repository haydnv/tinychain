import logging
import sys
import unittest

from .test_linalg import *
from .test_operators import *
from .test_table import *
from .test_tensor import *
from .test_app import *

logging.basicConfig(stream=sys.stderr)

unittest.main()
