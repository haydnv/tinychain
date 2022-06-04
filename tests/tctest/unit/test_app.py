import inspect
import logging
import sys
import unittest

import tinychain as tc

from .models import Order, Product, User

logger = logging.getLogger("test_app")


class Registry:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.models = dict()
        super().__init__()

    def register(self, model: tc.app.Model):
        """Adds a `Model` to the `models` attribute."""
        if model is None or (
            inspect.isclass(model) and not issubclass(model, tc.app.Model)
        ):
            logger.info("Model must be of type `tc.app.Model`")
            return

        if model.class_name() in self.models:
            raise ValueError(
                f"The `Model` {model.class_name()} has already been registered"
            )

        self.models[model.class_name()] = model


class App_(tc.app.App):
    def formulate(self):
        """Automatically build a Graph of all models associated with the app."""
        pass


class Arbitrary(tc.app.Model):
    """Dummy Model for the purpose of testing."""

    arbitrary_attribute = None

    def arbitrary_function(self):
        pass


class RegistryTests(unittest.TestCase):
    """Tests for the `App` class."""

    def tearDown(self):
        # Reset the singleton after every test run.
        Registry()._instance = None
        # Re-enable logging.
        logging.disable(logging.NOTSET)

    def test_Registry_singleton(self):
        self.assertIs(Registry(), Registry())

    def test_Registry_register_modelSubClass(self):
        registry = Registry()
        registry.register(User)

        SubClass = type("SubClass", (Arbitrary,), {})
        registry.register(SubClass)

        self.assertEqual(registry.models, {"user": User, "sub_class": SubClass})

    def test_Registry_register_modelNone(self):
        registry = Registry()
        registry.register(User)
        logging.disable(logging.INFO)  # info log is expected for this call
        registry.register(None)
        self.assertEqual(registry.models, {"user": User})

    def test_Registry_register_notModel(self):
        registry = Registry()
        registry.register(Product)
        logging.disable(logging.INFO)  # info log is expected for this call
        registry.register(dict)
        self.assertEqual(registry.models, {"product": Product})

    def test_Registry_register_alreadyRegistered(self):
        registry = Registry()
        registry.register(User)
        with self.assertRaises(ValueError):
            registry.register(User)


class AppTests(unittest.TestCase):
    """Tests for the `App` class."""

    def test_App_formulate(self):
        pass


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
