import unittest

from .base import HostTest


class ValueTests(HostTest):
    def testHelloWorld(self):
        self.assertEqual(self.host.get("/state/scalar/value/string", "Hello, World!"), "Hello, World!")


if __name__ == "__main__":
    unittest.main()
