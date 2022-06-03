import logging
import sys
import unittest

# from .test_linalg import *
# from .test_operators import *
# from .test_table import *
# from .test_tensor import *
# from .test_app import *
from .test_graph import *

logging.getLogger("test_graph")
logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

unittest.main()
