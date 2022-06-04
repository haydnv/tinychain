import logging
import sys
import unittest

from .test_app import *
from .test_graph import *

logging.basicConfig(level=logging.ERROR, stream=sys.stderr)

unittest.main()
