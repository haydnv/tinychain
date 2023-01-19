from collections import OrderedDict

from ...scalar.ref import deref, is_literal

from .base import NDArray


VALID_LABELS = set(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))


def parse_format(f):
    if is_literal(f):
        f = deref(f)
        if str(f) == f and "->" in f:
            f = str(f)
        else:
            raise ValueError(f"invalid format string for einsum: {f}")
    else:
        raise ValueError(f"einsum format string must be a compile-time constant, not {f}")

    f_inputs, f_output = f.split("->")

    if not f_inputs:
        raise ValueError(f"einsum format string {f} has no input format")

    f_inputs = [list(f) for f in f_inputs.split(',')]
    f_output = list(f_output)

    assert len(set(f_output)) == len(f_output)

    for f_input in f_inputs:
        if set(f_input) > VALID_LABELS:
            raise ValueError
        elif len(set(f_input)) < len(f_input):
            raise NotImplementedError(f"support for duplicate input axis labels in einsum: {f_input}")

    return f_inputs, f_output


def validate_args(f_inputs, tensors):
    if is_literal(f_inputs):
        f_inputs = deref(f_inputs)
    else:
        raise ValueError(f"einsum requires a literal format string, not {f_inputs}")

    if not hasattr(tensors, "__len__"):
        raise ValueError(f"einsum requires a literal number of tensors, not {tensors}")

    assert len(f_inputs) == len(tensors), f"the number of input formats {f_inputs} {len(f_inputs)} do not match tensors {tensors} ({len(tensors)})"

    dimensions = OrderedDict()
    for t in range(len(tensors)):
        fmt = f_inputs[t]
        assert tensors[t].ndim == len(fmt)

        for i in range(len(fmt)):
            if fmt[i] in dimensions:
                assert dimensions[fmt[i]] == tensors[t].shape[i]
            else:
                dimensions[fmt[i]] = tensors[t].shape[i]

    return dimensions


def outer_product(f_inputs, dimensions, tensors):
    assert is_literal(f_inputs)
    assert hasattr(tensors, "__len__")
    assert len(f_inputs) == len(tensors)

    f_output = list(dimensions.keys())
    tensors = list(tensors)

    regularized = []

    while tensors:
        tensor = tensors.pop()
        labels = f_inputs.pop()

        if labels == f_output:
            regularized.append(tensor)
            continue

        source = dict(zip(labels, range(len(labels))))
        permutation = [source[l] for l in f_output if l in labels]
        labels = [labels[axis] for axis in permutation]
        tensor = tensor.transpose(permutation)

        # TODO: replace tensor.expand_dims with tensor.unsqueeze in the calculation below
        i = 0
        while i < len(dimensions):
            if i == len(labels) or labels[i] != f_output[i]:
                tensor = tensor.expand_dims(i)
                labels.insert(i, f_output[i])
            else:
                i += 1

        assert deref(tensor.ndim) == len(f_output)
        regularized.append(tensor)

    op = regularized.pop()
    while regularized:
        tensor = regularized.pop()
        op = op * tensor

    return op


def contract(op, dimensions, f_output):
    assert is_literal(f_output)

    if not f_output:
        return op.sum()

    f_input = list(dimensions.keys())
    axis = 0
    while len(f_input) > len(f_output):
        if f_input[axis] not in f_output:
            op = op.sum(axis)
            del f_input[axis]
        else:
            axis += 1

    if f_input == f_output:
        return op
    else:
        source = dict(zip(f_input, range(len(f_input))))
        permutation = [source[l] for l in f_output]
        return op.transpose(permutation)


def einsum(f, tensors):
    """
    Return the Einstein summation of the given `tensors` according the the given format string.

    Example: `einsum("ij,jk->ik", [A, B])  # multiply matrices A and B`

    The tensor product is computed from left to right, so when using any `Sparse` tensors,
    it's important to put the sparsest first in the list to avoid redundant broadcasting.
    """

    if not is_literal(f):
        raise ValueError(f"einsum requires a literal format, not {format}")

    if not hasattr(tensors, "__iter__"):
        raise ValueError(f"einsum requires a literal number of tensors, not {tensors}")

    for tensor in tensors:
        if not isinstance(tensor, NDArray):
            raise TypeError(f"einsum requires a tensor, not: {tensor}")

    f_inputs, f_output = parse_format(f)
    dimensions = validate_args(f_inputs, tensors)

    op = outer_product(f_inputs, dimensions, tensors)
    # assert is_literal(op.ndim), is_literal(op.shape)
    # assert deref(op.shape) == tuple(dimensions.values())
    return contract(op, dimensions, f_output)
