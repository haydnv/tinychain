from __future__ import annotations

import typing as t

import numpy as np


class GraphNode:
    name: str

    def __init__(self, value: Tensor) -> None:
        self._value = value

    @property
    def value(self) -> Tensor:
        return self._value

    def backward(self) -> Tensor:
        raise NotImplementedError()


class TensorPlaceholderGraphNode(GraphNode):
    name: str = 'tensor_placeholder'

    def backward(self) -> Tensor:
        return self._value


class TensorAddGraphNode(GraphNode):
    name: str = 'tensor_add'

    def __init__(self, value: Tensor, left: GraphNode, right: GraphNode) -> None:
        super().__init__(value)
        self._left = left
        self._right = right

    def backward(self) -> Tensor:
        return self._left.backward() + self._right.backward()


class TensorSubGraphNode(GraphNode):
    name: str = 'tensor_sub'

    def __init__(self, value: Tensor, left: GraphNode, right: GraphNode) -> None:
        super().__init__(value)
        self._left = left
        self._right = right

    def backward(self) -> Tensor:
        return self._left.backward() - self._right.backward()


class TensorMulGraphNode(GraphNode):
    name: str = 'tensor_mul'

    def __init__(self, value: Tensor, left: GraphNode, right: GraphNode) -> None:
        super().__init__(value)
        self._left = left
        self._right = right

    def backward(self) -> Tensor:
        return self._left.backward() * self._right.value + self._left.value * self._right.backward()

#TODO: Implementanit Tensor.__pow__()
class TensorPowGraphNode(GraphNode):
    name: str = 'tensor_pow'

    def __init__(self, value: Tensor, left: GraphNode, right: GraphNode) -> None:
        super().__init__(value)
        self._left = left
        self._right = right

    # def backward(self) -> Tensor:
    #     return self._right.value * (self._left.value**(self._right.value - Tensor(1)))


class GradientTape:
    _current: t.Optional[GradientTape] = None

    def __init__(self) -> None:
        self._node_dict: t.Dict[int, GraphNode] = {}

    @classmethod
    def get_current_tape(cls) -> GradientTape:
        if cls._current is None:
            cls._current = cls()
        return cls._current

    def track(self, id: int, node: GraphNode):
        self._node_dict[id] = node

    def get(self, id: int) -> t.Optional[GraphNode]:
        return self._node_dict.get(id)

    def get_or_raise(self, id: int) -> GraphNode:
        node = self.get(id)
        assert node is not None, 'Operand should have gradient.'
        return node


class Tensor:

    def __init__(self, inner: np.ndarray):
        self._inner = inner

    def __add__(self, other: Tensor) -> Tensor:
        result = Tensor(self._inner.__add__(other._inner))
        tape = GradientTape.get_current_tape()
        tape.track(id(result), TensorAddGraphNode(
            result,
            tape.get_or_raise(id(self)),
            tape.get_or_raise(id(other)),
        ))
        return result

    def __sub__(self, other: Tensor) -> Tensor:
        result = Tensor(self._inner.__sub__(other._inner))
        tape = GradientTape.get_current_tape()
        tape.track(id(result), TensorSubGraphNode(
            result,
            tape.get_or_raise(id(self)),
            tape.get_or_raise(id(other)),
        ))
        return result

    def __mul__(self, other: Tensor) -> Tensor:
        result = Tensor(self._inner.__mul__(other._inner))
        tape = GradientTape.get_current_tape()
        tape.track(id(result), TensorMulGraphNode(
            result,
            tape.get_or_raise(id(self)),
            tape.get_or_raise(id(other)),
        ))
        return result

    def __pow__(self, other) -> Tensor:
        result = Tensor(self._inner.__pow__(other._inner))
        tape = GradientTape.get_current_tape()
        tape.track(id(result), TensorPowGraphNode(
            result,
            tape.get_or_raise(id(self)),
            tape.get_or_raise(id(other)),
        ))
        return result

    def enable_grad(self):
        tape = GradientTape.get_current_tape()
        tape.track(id(self), TensorPlaceholderGraphNode(self))

    def backward(self) -> Tensor:
        tape = GradientTape.get_current_tape()
        return tape.get_or_raise(id(self)).backward()

    def to_numpy(self) -> np.ndarray:
        return self._inner


def tensor(*args, enable_grad: bool = False, **kwargs) -> Tensor:
    ten = Tensor(np.array(*args, **kwargs))
    if enable_grad:
        ten.enable_grad()
    return ten


a = tensor(np.array([1.1, 0, 0, 1.1]).reshape((2, 2)), enable_grad=True)
b = tensor(np.array([1, 0, 0, 1]).reshape((2, 2)), enable_grad=True)
c = tensor(np.array([2, 1, 4, 3]).reshape((2, 2)), enable_grad=True)

p = tensor(np.array([2]), enable_grad=True)

k = a + b + b * c
j = k + a + a + c * k
l = a**p
print(l.backward().to_numpy())
