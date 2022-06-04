import logging
import sys
import unittest

from .test_app import *
from .test_graph import *

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

unittest.main()
