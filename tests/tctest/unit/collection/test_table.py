import logging
import unittest

import tinychain as tc

from ..models import Order, Product, User

logger = logging.getLogger("test_tablw")


class Arbitrary(tc.app.Model):
    """Dummy Model for the purpose of testing."""

    arbitrary_attribute = None

    def arbitrary_function(self):
        pass


class TableCreateSchemaTests(unittest.TestCase):
    """Tests for the `create_schema` function."""

    def test_createSchema_withArbitraryValues(self):
        """Test that creating a schema ignores arbitrary attributes. Only
        values of Column or Model are recognised.
        """
        schema = tc.table.create_schema(Arbitrary)
        expected = tc.table.Schema([tc.Column("arbitrary_id", tc.U32)], [])
        self.assertIsInstance(schema, tc.table.Schema)
        self.assertEqual(
            sorted(schema.columns(), key=str), sorted(expected.columns(), key=str)
        )

    def test_createSchema_simple(self):
        """Test that creating a schema works using a basic Model."""
        schema = tc.table.create_schema(User)
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
        schema = tc.table.create_schema(Order)
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
