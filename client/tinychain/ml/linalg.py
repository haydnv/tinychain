from tinychain.collection.tensor import einsum, Dense, Schema, Sparse, Tensor
from tinychain.decorators import closure, get_op, post_op
from tinychain.ref import After, Get, If, MethodSubject, While
from tinychain.state import Map, Stream, Tuple
from tinychain.value import Bool, F64, Float, UInt

# from "Numerical Recipes in C" p. 65
EPS = 10**-6


def diagonal(matrix):
    """Return the diagonal of the given `matrix`"""

    rtype = type(matrix) if isinstance(matrix, Tensor) else Tensor
    op = Get(MethodSubject(matrix, "diagonal"))
    return rtype(op)


def identity(size, dtype=Bool):
    """Return an identity matrix with dimensions `[size, size]`."""

    schema = Schema([size, size], dtype)
    elements = Stream.range((0, size)).map(get_op(lambda i: ((i, i), 1)))
    return Sparse.copy_from(schema, elements)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
@post_op
def householder(cxt, x: Tensor) -> Tuple:
    """Compute the Householder vector of the given column vector `a`."""

    cxt.alpha = x[0]
    cxt.s = (x[1:]**2).sum()
    cxt.t = (cxt.alpha**2 + cxt.s)**0.5

    cxt.v = x.copy()  # make a copy in case X is updated before the return values are evaluated
    cxt.v_zero = F64(If(cxt.alpha <= 0, cxt.alpha - cxt.t, -cxt.s / (cxt.alpha + cxt.t)))
    tau = If(cxt.s.abs() < EPS, 0, 2 * cxt.v_zero**2 / (cxt.s + cxt.v_zero ** 2))
    v = After(cxt.v[0].write(cxt.v_zero), cxt.v / cxt.v_zero)

    return v, tau


def matmul(l: Tensor, r: Tensor):
    """
    Multiply two matrices, or two batches of matrices.

    Args:
        `l`: a `Tensor` with shape `[..., i, j]`
        `r`: a `Tensor` with shape `[..., j, k]`

    Returns:
        A `Tensor` of shape `[..., i, k]`
    """

    return einsum("...ij,...jk->ik", [l, r])


def norm(tensor: Tensor) -> Tensor:
    """Compute the 2D Frobenius (aka Euclidean) norm of the matrices in the given `tensor`.

    Args:
        `tensor`: a matrix or batch of matrices with shape `[..., M, N]`

    Returns:
        A `Tensor` of shape `[...]` or a `Number` if the input `tensor` is itself 2-dimensional
    """

    squared = tensor**2
    return If(tensor.ndim <= 2,
              squared.sum()**0.5,
              squared.sum(-1).sum(-1)**0.5)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
@post_op
def qr(cxt, x: Tensor) -> Tuple:
    """Compute the QR factorization of the given `matrix`.

    Args:
        `a`: a matrix with shape `[M, N]`

    Returns:
        A `Tuple` of `Tensor`s `(Q, R)` where `A ~= QR` and `Q.transpose() == Q**-1`
    """

    cxt.m = UInt(x.shape[0])
    cxt.n = UInt(x.shape[1])
    cxt.householder = householder

    outer_cxt = cxt

    @closure
    @post_op
    def qr_step(cxt, Q: Tensor, R: Tensor, k: UInt) -> Map:
        cxt.v, cxt.tau = outer_cxt.householder(x=R[k:, k]).unpack(2)
        cxt.v_outer = einsum("i,j->ij", [cxt.v, cxt.v])

        cxt.H = identity(outer_cxt.m, F64).as_dense().copy()
        cxt.H_sub = (cxt.H[k:, k:] - (cxt.v_outer * cxt.tau))
        return After(cxt.H[k:, k:].write(cxt.H_sub), {
            "Q": matmul(cxt.H, Q),
            "R": matmul(cxt.H, R),
        })

    state = Map(Q=identity(cxt.m, F64).as_dense(), R=x)
    QR = Stream.range(cxt.n - 1).fold("k", state, qr_step)
    return Tensor(QR['Q'])[:cxt.n].transpose(), Tensor(QR['R'])[:cxt.n]


@post_op
def bidiagonalize(cxt, x: Tensor) -> Tuple:
    cxt.m = UInt(x.shape[0])
    cxt.n = UInt(x.shape[1])

    cxt.householder = householder

    outer_cxt = cxt

    @closure
    @post_op
    def left(cxt, k: UInt, A: Tensor, U: Tensor) -> Map:
        cxt.v, cxt.tau = outer_cxt.householder(x=A[k:, k]).unpack(2)
        cxt.v_outer_tau = einsum("i,j->ij", [cxt.v, cxt.v]) * cxt.tau

        diagonal = identity(outer_cxt.m - k) - cxt.v_outer_tau
        diagonal = matmul(diagonal, A[k:, k:])
        A = After(A[k:, k:].write(diagonal), A)

        cxt.I_m = identity(outer_cxt.m, F64).as_dense().copy()
        Q_k = After(cxt.I_m[k:, k:].write(cxt.v_outer_tau), cxt.I_m)
        U = matmul(Q_k, U)

        return {"U": U, "A": A}

    @closure
    @post_op
    def right(cxt, k: UInt, A: Tensor, U: Tensor, V: Tensor) -> Map:
        cxt.v, cxt.tau = outer_cxt.householder(x=A[k, k + 1:]).unpack(2)
        cxt.v_outer_tau = einsum("i,j->ij", [cxt.v, cxt.v]) * cxt.tau

        diagonal = identity(outer_cxt.n - (k + 1)) - cxt.v_outer_tau
        diagonal = matmul(A[k:, k + 1:], diagonal)
        A = After(A[k:, k + 1:].write(diagonal), A)

        cxt.I_n = identity(outer_cxt.n, F64).as_dense().copy()
        P = After(cxt.I_n[k + 1:, k + 1:].write(cxt.v_outer_tau), cxt.I_n)
        V = matmul(P, V)

        return {"U": U, "A": A, "V": V}

    cxt.left = left
    cxt.right = right

    @closure
    @post_op
    def step(A: Tensor, U: Tensor, V: Tensor, k: UInt) -> Map:
        left = cxt.left(k=k, A=A, U=U)
        right = cxt.right(k=k, A=left["A"], U=left["U"], V=V)
        return If(k < cxt.n - 2, right, left)

    U = identity(cxt.m, F64).as_dense()
    V = identity(cxt.n, F64).as_dense()

    return Stream.range(cxt.n - 2).fold("k", Map(A=x.copy(), U=U, V=V), step)


def svd(matrix: Tensor) -> Tuple:
    """Return the singular value decomposition of the given `matrix`."""

    rtype = type(matrix) if isinstance(matrix, Tensor) else Tensor
    op = Get(MethodSubject(matrix, "svd"))
    return rtype(op)
