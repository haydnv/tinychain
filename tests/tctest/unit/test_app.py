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
            (type("HiAA", (tc.app.Model,), {}), "hi_a_a"),
        ]
        for c, e in cases:
            with self.subTest(c=c, e=e):
                self.assertEqual(tc.app.class_name(c), e)

    def test_key(self):
        """Parameterized unit test for the `key` function."""
        cases = [
            (User, [tc.Column("user_id", tc.U32)]),
            (Order, [tc.Column("order_id", tc.U32)]),
            (Product, [tc.Column("product_id", tc.U32)]),
        ]
        for c, e in cases:
            with self.subTest(c=c, e=e):
                self.assertEqual(c.key(), e)
