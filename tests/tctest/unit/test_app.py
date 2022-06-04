import unittest

import tinychain as tc

from .configure import Order, Product, User


class Arbitrary(tc.app.Model):
    """Dummy Model for the purpose of testing."""

    arbitrary_attribute = None

    def arbitrary_function(self):
        pass


class AppTests(unittest.TestCase):
    """Tests for the `App` class."""
    
    def test_App_formulate(self):
        self.assertEquals(0, 1)


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


class CreateSchemaTests(unittest.TestCase):
    """Tests for the `create_schema` function."""

    def test_createSchema_withArbitraryValues(self):
        """Test that creating a schema ignores arbitrary attributes. Only
        values of Column or Model are recognised.
        """
        schema = tc.app.create_schema(Arbitrary)
        expected = tc.table.Schema([tc.Column("arbitrary_id", tc.U32)], [])
        self.assertIsInstance(schema, tc.table.Schema)
        self.assertEqual(
            sorted(schema.columns(), key=str), sorted(expected.columns(), key=str)
        )

    def test_createSchema_simple(self):
        """Test that creating a schema works using a basic Model."""
        schema = tc.app.create_schema(User)
        expected = tc.table.Schema(
            [tc.Column("user_id", tc.U32)],
            [
                tc.Column("first_name", tc.String, 100),
                tc.Column("last_name", tc.String, 100),
            ],
        )
        self.assertIsInstance(schema, tc.table.Schema)
        self.assertEqual(
            sorted(schema.columns(), key=str), sorted(expected.columns(), key=str)
        )

    def test_createSchema_complex(self):
        """Test that creating a schema works using a complex Model."""
        schema = tc.app.create_schema(Order)
        expected = (
            tc.table.Schema(
                [tc.Column("order_id", tc.U32)],
                [
                    tc.Column("product_id", tc.U32),
                    tc.Column("user_id", tc.U32),
                    tc.Column("quantity", tc.I32),
                ],
            )
            .create_index("user", ["user_id"])
            .create_index("product", ["product_id"])
        )
        self.assertIsInstance(schema, tc.table.Schema)
        self.assertEqual(sorted(schema.indices), sorted(expected.indices))
        self.assertEqual(
            sorted(schema.columns(), key=str), sorted(expected.columns(), key=str)
        )
