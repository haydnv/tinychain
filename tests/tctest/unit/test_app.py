import logging
import unittest

import tinychain as tc

from .models import Order, Product, User

logger = logging.getLogger("test_app")


class ModelTests(unittest.TestCase):
    """Tests for the `Model` class."""

    def test_className(self):
        """Parameterized unit test for the `class_name` function."""
        cases = [
            (User, "user"),
            (User(1, "first", "last"), "user"),
            (type("HiAA", (tc.service.Model,), {}), "hi_a_a"),
        ]
        for c, e in cases:
            with self.subTest(c=c, e=e):
                self.assertEqual(tc.service.class_name(c), e)
