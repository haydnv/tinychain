import os

from tinychain.util import URI


LIB_URI = URI(os.getenv("TC_URI", "/lib/ml"))
