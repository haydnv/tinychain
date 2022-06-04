import logging
import unittest

import tinychain as tc

from .models import Product, User

logger = logging.getLogger("test_registry")


class RegistryTests(unittest.TestCase):
    """Tests for the `Registry` class."""

    def setUp(self):
        # Reset the singleton after every test run.
        self.registry = tc.registry.Registry(create_new=True)

    def tearDown(self):
        # Re-enable logging.
        logging.disable(logging.NOTSET)

    def test_Registry_singleton(self):
        self.assertIs(tc.registry.Registry(), self.registry)

    def test_Registry_register_modelSubClass(self):
        self.registry.register(User)

        SubClassUser = type("SubClassUser", (User,), {})
        self.registry.register(SubClassUser)

        self.assertEqual(
            self.registry.models, {"user": User, "sub_class_user": SubClassUser}
        )

    def test_Registry_register_modelNone(self):
        self.registry.register(User)
        logging.disable(logging.INFO)  # info log is expected for this call

        self.registry.register(None)

        self.assertEqual(self.registry.models, {"user": User})

    def test_Registry_register_notModel(self):
        self.registry.register(Product)
        logging.disable(logging.INFO)  # info log is expected for this call

        self.registry.register(dict)

        self.assertEqual(self.registry.models, {"product": Product})

    def test_Registry_register_alreadyRegistered(self):
        self.registry.register(User)
        with self.assertRaises(ValueError):
            self.registry.register(User)
