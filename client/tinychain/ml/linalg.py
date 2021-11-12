from tinychain.collection.tensor import einsum, Dense, Schema, Sparse, Tensor
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
    """Compute the Householder vector of the given column `a`."""

    cxt.a = x.copy()  # make a copy in case X is updated before the return values are evaluated
    cxt.a_norm = (cxt.a**2).sum()**0.5
    cxt.v = (cxt.a / (cxt.a[0] + cxt.a_norm)).copy()
    tau = 2 / (cxt.v**2).sum()

    return Tuple(After(cxt.v.write([0], 1), (cxt.v, tau)))


def norm(tensor: Tensor) -> Tensor:
    """Compute the 2D Frobenius (aka Euclidean) norm of the matrices in the given `tensor`.

    Args:
        `tensor`: a matrix or batch of matrices with shape `[..., M, N]`

    Returns:
        A `Tensor` of shape [...] or a `Number` if the input `tensor` is itself 2-dimensional
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
        cxt.transform = outer_cxt.householder(x=R[k:, k])
        cxt.v_outer = einsum("i,j->ij", [cxt.transform[0], cxt.transform[0]])
        cxt.tau = F32(cxt.transform[1])

        cxt.H = identity(outer_cxt.m, F32).as_dense().copy()
        cxt.H_sub = (cxt.H[k:, k:] - (cxt.v_outer * cxt.tau))
        return After(cxt.H.write([slice(k, None), slice(k, None)], cxt.H_sub), {
            "Q": einsum("ij,jk->ik", [cxt.H, Q]),
            "R": einsum("ij,jk->ik", [cxt.H, R]),
        })

    state = Map(Q=identity(cxt.m, F32).as_dense(), R=x)
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
        cxt.transform = outer_cxt.householder(x=A[k:, k])
        cxt.v_outer_tau = einsum("i,j->ij", [cxt.transform[0], cxt.transform[0]]) * cxt.transform[1]

        diagonal = identity(outer_cxt.m - k) - cxt.v_outer_tau
        diagonal = einsum("ij,jk->ik", [diagonal, A[k:, k:]])
        A = After(A.write([slice(k, None), slice(k, None)], diagonal), A)

        cxt.I_m = identity(outer_cxt.m, F32).as_dense().copy()
        Q_k = After(cxt.I_m.write([slice(k, None), slice(k, None)], cxt.v_outer_tau), cxt.I_m)
        U = einsum("ij,jk->ik", [Q_k, U])

        return {"U": U, "A": A}

    @closure
    @post_op
    def right(cxt, k: UInt, A: Tensor, U: Tensor, V_t: Tensor) -> Map:
        cxt.transform = outer_cxt.householder(x=A[k, k + 1:])
        cxt.v_outer_tau = einsum("i,j->ij", [cxt.transform[0], cxt.transform[0]]) * cxt.transform[1]

        diagonal = identity(outer_cxt.n - (k + 1)) - cxt.v_outer_tau
        diagonal = einsum("ij,jk->ik", [A[k:, k + 1:], diagonal])
        A = After(A.write([slice(k, None), slice(k + 1, None)], diagonal), A)

        cxt.I_n = identity(outer_cxt.n, F32).as_dense().copy()
        P = After(cxt.I_n.write([slice(k + 1, None), slice(k + 1, None)], cxt.v_outer_tau), cxt.I_n)
        V_t = einsum("ij,jk->ik", [P, V_t])

        return {"U": U, "A": A, "V_t": V_t}

    cxt.left = left
    cxt.right = right

    @closure
    @post_op
    def step(A: Tensor, U: Tensor, V_t: Tensor, k: UInt) -> Map:
        left = cxt.left(k=k, A=A, U=U)
        right = cxt.right(k=k, A=left["A"], U=left["U"], V_t=V_t)
        return If(k < cxt.n - 2, right, left)

    U = identity(cxt.m, F32).as_dense()
    V_t = identity(cxt.n, F32).as_dense()

    return Stream.range(cxt.n - 2).fold("k", Map(A=x, U=U, V_t=V_t), step)
