import inspect
import typing

from ..app import Model, ModelRef
from ..collection import tensor
from ..generic import Map, Tuple
from ..scalar.number import Number
from ..scalar.ref import form_of


DType = typing.TypeVar("DType", bound=type[Number])


# TODO: support Sparse and Number variable types
class Variable(tensor.Dense, typing.Generic[DType]):
    """A trainable variable in a machine learning model."""

    def update(self, delta):
        """Decrement the value of this `Variable` by `delta`."""

        return self.write(self - delta)


def namespace(model, prefix=None):
    """Traverse the attributes of the given `model` to describe the namespace of its trainable :class:`Variable` s."""

    def suffix(name):
        if prefix:
            return f"{prefix}.{name}"
        else:
            return str(name)

    if isinstance(model, Variable):
        return {prefix: model}
    elif isinstance(model, ModelRef):
        return namespace(model.instance, prefix)

    if isinstance(model, (Map, Tuple)):
        model = form_of(model)

    ns = {}

    if isinstance(model, (list, tuple)):
        for i, component in enumerate(model):
            ns.update(namespace(component, suffix(i)))
    elif isinstance(model, dict):
        for name, component in model.items():
            ns.update(namespace(component, suffix(name)))
    elif isinstance(model, Model):
        for name, component in inspect.getmembers(model):
            if name.startswith("__"):
                continue

            ns.update(namespace(component, suffix(name)))

    return ns
