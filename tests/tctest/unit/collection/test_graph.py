import logging
import unittest

import tinychain as tc

from ..models import Order, Product, User

logger = logging.getLogger("test_graph")


class Graph_(tc.graph.Graph):
    # Needed to avoid a RecursionError.
    __uri__ = tc.URI("/test/unit/graph")


class GraphTests(unittest.TestCase):

    def test_initaliseSchema_graphCreatedOnInit(self):
        tc.registry.Registry(create_new=True).register(User)
        expected = tc.graph.Schema().create_table("user", tc.table.create_schema(User))
        graph = Graph_()
        self.assertIsInstance(graph.schema, tc.graph.Schema)
        self.assertEqual(sorted(graph.schema.tables), sorted(expected.tables))


class GraphCreateSchemaTests(unittest.TestCase):

    def test_createSchema_simple(self):
        """Test that creating a schema works using a basic Model."""
        users = tc.table.create_schema(User)
        products = tc.table.create_schema(Product)
        expected = (
            tc.graph.Schema()
            .create_table("user", users)
            .create_table("product", products)
        )
        schema = tc.graph.create_schema([users, products])
        self.assertIsInstance(schema, tc.graph.Schema)
        self.assertEqual(sorted(schema.tables), sorted(expected.tables))
        self.assertEqual(sorted(schema.edges), sorted(expected.edges))

    def test_createSchema_complex(self):
        """Test that creating a schema works using a complex Model that
        has relations."""
        users = tc.table.create_schema(User)
        orders = tc.table.create_schema(Order)
        products = tc.table.create_schema(Product)
        expected = (
            tc.graph.Schema()
            .create_table("user", users)
            .create_table("product", products)
            .create_table("order", orders)
            .create_edge(
                "order_product",
                tc.graph.edge.Schema("product.product_id", "order.product_id"),
            )
            .create_edge(
                "order_user", tc.graph.edge.Schema("user.user_id", "order.user_id")
            )
        )
        schema = tc.graph.create_schema([users, products, orders])
        self.assertIsInstance(schema, tc.graph.Schema)
        self.assertEqual(sorted(schema.tables), sorted(expected.tables))
        self.assertEqual(sorted(schema.edges), sorted(expected.edges))
