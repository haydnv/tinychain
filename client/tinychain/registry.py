import inspect
import logging
import sys
import unittest

from .app import Model

logger = logging.getLogger(__name__)


class Registry:

    models = None
    _instance = None

    def __new__(cls, create_new=False):
        """Using the `create_new` kwarg we can force recreate the singleton if it is 
        set to `True`.
        """
        if not isinstance(cls._instance, cls) or create_new:
            cls._instance = object.__new__(cls)
            cls.models = dict()
        return cls._instance

    def register(self, model: Model):
        """Adds a `Model` to the `models` attribute."""
        if model is None or (
            inspect.isclass(model) and not issubclass(model, Model)
        ):
            logger.info("Model must be of type `tc.app.Model`")
            return

        if model.class_name() in self.models:
            raise ValueError(
                f"The `Model` {model.class_name()} has already been registered"
            )

        self.models[model.class_name()] = model
