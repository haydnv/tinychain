from tinychain.collection.tensor import einsum, Schema, Sparse, Tensor
from tinychain.decorators import closure, get_op, post_op
from tinychain.ref import After, If
from tinychain.state import Map, Stream, Tuple
from tinychain.value import Bool, F32, Float, UInt


def identity(size, dtype=Bool):
    """Return an identity matrix with dimensions `[size, size]`."""

    schema = Schema([size, size], dtype)
    elements = Stream.range((0, size)).map(get_op(lambda i: ((i, i), 1)))
    return Sparse.copy_from(schema, elements)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
@post_op
def householder(cxt, x: Tensor) -> Tuple:
    """Computes Householder vector for `a`."""

    a = x.copy()
    cxt.v = (a / (a[0] + norm(a))).copy()
    tau = (2 / einsum("ji,jk->ik", [cxt.v, cxt.v]))

    return Tuple(After(cxt.v.write([0], 1), (cxt.v, tau)))


def norm(tensor: Tensor) -> Tensor:
    """Compute the 2D Frobenius (aka Euclidean) norm of the matrices in the given `tensor`.

    Args:
        `tensor`: a matrix or batch of matrices with shape `[..., M, N]`

    Returns:
        A `Tensor` of shape [...] or a `Number` if the input `tensor` is itself 2-dimensional
    """

    squared = tensor**2
    return If(tensor.ndim == 2,
              squared.sum()**0.5,
              squared.sum(-1).sum(-1)**0.5)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
# TODO: handle rectangular matrices
@post_op
def qr(cxt, matrix: Tensor) -> Tuple:
    """Compute the QR factorization of the given `matrix`.

    Args:
        `a`: a matrix with shape `[M, N]`

    Returns:
        A `Tuple` of `Tensor`s `(Q, R)` where `A ~= QR` and `Q.transpose() == Q**-1`
    """

    cxt.m = UInt(matrix.shape[0])
    cxt.n = UInt(matrix.shape[1])
    cxt.householder = householder

    outer_cxt = cxt

    @closure
    @post_op
    def qr_step(cxt, Q: Tensor, R: Tensor, k: UInt) -> Map:
        cxt.column = R[k:, k].expand_dims()
        cxt.transform = outer_cxt.householder(x=cxt.column)
        v = Tensor(cxt.transform[0])
        tau = Tensor(cxt.transform[1])

        cxt.H = identity(outer_cxt.m, F32).as_dense().copy()
        cxt.H_sub = (cxt.H[k:, k:] - (tau * einsum("ij,kj->ik", [v, v])))
        return After(cxt.H.write([slice(k, None), slice(k, None)], cxt.H_sub), {
            "Q": einsum("ij,jk->ik", [cxt.H, Q]),
            "R": einsum("ij,jk->ik", [cxt.H, R]),
        })

    QR = Stream.range(cxt.n).fold("k", Map(Q=identity(cxt.m, F32).as_dense(), R=matrix.copy()), qr_step)
    return Tensor(QR['Q'])[:cxt.n].transpose(), Tensor(QR['R'])[:cxt.n]
