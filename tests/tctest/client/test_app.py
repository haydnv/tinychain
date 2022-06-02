import unittest

import tinychain as tc

from .base import ClientTest

URI = tc.URI("/test/app")


class Product(tc.app.Model):
    __uri__ = URI.append("Product")

    price = tc.Column("price", tc.I32)
    name = tc.Column("name", tc.String, 100)

    def __init__(self, product, name, price):
        self.product = product
        self.price = price
        self.name = name


class User(tc.app.Model):
    __uri__ = URI.append("User")

    first_name = tc.Column("first_name", tc.String, 100)
    last_name = tc.Column("last_name", tc.String, 100)

    def __init__(self, user_id, first_name, last_name):
        self.user_id = tc.Column("user_id", tc.I32)
        self.first_name = first_name
        self.last_name = last_name


class Order(tc.app.Model):
    __uri__ = URI.append("Product")

    quantity = tc.Column("quantity", tc.U32)
    product_id = Product
    user_id = User

    def __init__(self, order_id, quantity, user_id, product_id):
        self.quantity = quantity
        self.user_id = user_id
        self.product_id = product_id


class Arbitrary(tc.app.Model):
    """Dummy Model for the purpose of testing."""

    arbitrary_attribute = None

    def arbitrary_function(self):
        pass


class ModelTests(ClientTest):

    def testModelClassName(self):
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

    def testModelKey(self):
        """Parameterized unit test for the `key` function."""
        cases = [
            (User, ["user_id", tc.I32]),
            (Order, ["order_id", tc.I32]),
            (Product, ["product_id", tc.I32]),
        ]
        for c, e in cases:
            with self.subTest(c=c, e=e):
                self.assertEqual(c.key(), e)

    def testCreateSchemaWithArbitraryValues(self):
        """Test that creating a schema ignores arbitrary attributes. Only
        values of Column or Model are recognised.
        """
        schema = Arbitrary.create_schema()
        expected = tc.table.Schema(["arbitrary_id", tc.I32], [])
        self.assertIsInstance(schema, tc.table.Schema)
        self.assertEqual(
            sorted(schema.columns(), key=str), sorted(expected.columns(), key=str)
        )

    def testCreateSchemaSimple(self):
        """Test that creating a schema works using a basic Model."""
        schema = User.create_schema()
        expected = tc.table.Schema(
            ["user_id", tc.I32],
            [
                tc.Column("first_name", tc.String, 100),
                tc.Column("last_name", tc.String, 100),
            ],
        )
        self.assertIsInstance(schema, tc.table.Schema)
        self.assertEqual(
            sorted(schema.columns(), key=str), sorted(expected.columns(), key=str)
        )

    def testCreateSchemaComplex(self):
        """Test that creating a schema works using a complex Model."""
        schema = Order.create_schema()
        expected = (
            tc.table.Schema(
                ["order_id", tc.I32],
                [
                    tc.Column("product_id", tc.I32),
                    tc.Column("user_id", tc.I32),
                    tc.Column("quantity", tc.U32),
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
