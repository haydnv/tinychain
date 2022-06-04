import logging
import unittest

import tinychain as tc

from .models import Order, Product, User

logger = logging.getLogger("test_app")


class ModelTests(unittest.TestCase):
    """Tests for the `Model` class."""

    def test_Model_className(self):
        """Parameterized unit test for the `class_name` function."""
        cases = [
            (User, "user"),
            (Order, "order"),
            (Product, "product"),
            (type("A", (tc.app.Model,), {}), "a"),
            (type("AA", (tc.app.Model,), {}), "a_a"),
            (type("HiAA", (tc.app.Model,), {}), "hi_a_a"),
        ]
        for c, e in cases:
            with self.subTest(c=c, e=e):
                self.assertEqual(c.class_name(), e)

    def test_Model_key(self):
        """Parameterized unit test for the `key` function."""
        cases = [
            (User, [tc.Column("user_id", tc.U32)]),
            (Order, [tc.Column("order_id", tc.U32)]),
            (Product, [tc.Column("product_id", tc.U32)]),
        ]
        for c, e in cases:
            with self.subTest(c=c, e=e):
                self.assertEqual(c.key(), e)
