import unittest

import tinychain as tc

from .base import ClientTest
from .configure import Order, Product, User


class GraphTests(ClientTest):
    def testCreateSchemaSimple(self):
        """Test that creating a schema works using a basic Model."""
        users = tc.app.create_schema(User)
        products = tc.app.create_schema(Product)
        expected = (
            tc.graph.Schema()
            .create_table("user", users)
            .create_table("product", products)
        )
        schema = tc.graph.create_schema([users, products])
        self.assertIsInstance(schema, tc.graph.Schema)
        self.assertEqual(sorted(schema.tables), sorted(expected.tables))
        self.assertEqual(sorted(schema.edges), sorted(expected.edges))

    def testCreateSchemaComplex(self):
        """Test that creating a schema works using a basic Model."""
        users = tc.app.create_schema(User)
        orders = tc.app.create_schema(Order)
        products = tc.app.create_schema(Product)
        expected = (
            tc.graph.Schema()
            .create_table("user", users)
            .create_table("product", products)
            .create_table("order", orders)
            .create_edge("order_product", tc.graph.edge.Schema("product.product_id", "order.product_id"))
            .create_edge("order_user", tc.graph.edge.Schema("user.user_id", "order.user_id"))
        )
        schema = tc.graph.create_schema([users, products, orders])
        self.assertIsInstance(schema, tc.graph.Schema)
        self.assertEqual(sorted(schema.tables), sorted(expected.tables))
        self.assertEqual(sorted(schema.edges), sorted(expected.edges))
