import logging
import sys
import unittest

from .collection.test_graph import *
from .collection.test_table import *
from .test_app import *

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

unittest.main()
