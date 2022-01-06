from tinychain.collection.tensor import einsum, Dense, Schema, Sparse, Tensor
from tinychain.decorators import closure, get_op, post_op
from tinychain.ref import After, Get, If, MethodSubject, While
from tinychain.state import Map, Stream, Tuple
from tinychain.value import Bool, F64, Float, UInt, F32

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


def set_diagonal(matrix, diag):
    """Set the diagonal of the given `matrix` to `diag`."""

    eye = identity(matrix.shape[0])
    # don't use eye.logical_not in case the matrix is sparse
    zero_diag = matrix - (matrix * eye)
    new_diag = eye * diag.expand_dims()
    return matrix.write(zero_diag + new_diag)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
@post_op
def householder(cxt, x: Tensor) -> Tuple:
    """Compute the Householder vector of the given column vector `a`."""

    cxt.alpha = x[0]
    cxt.s = (x[1:]**2).sum()
    cxt.t = (cxt.alpha**2 + cxt.s)**0.5

    cxt.v = x.copy()  # make a copy in case X is updated before the return values are evaluated
    cxt.v_zero = F64(If(cxt.alpha <= 0, cxt.alpha -
                     cxt.t, -cxt.s / (cxt.alpha + cxt.t)))
    tau = If(cxt.s.abs() < EPS, 0, 2 * cxt.v_zero **
             2 / (cxt.s + cxt.v_zero ** 2))
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


class PLUFactorization(Map):
    """
    PLU factorization of a given `[N, N]` matrix.
    """

    @property
    def p(self) -> Tensor:
        """
        Permutation matrix as an `[N, N]` `Tensor`.
        """
        return Tensor(self['p'])

    @property
    def l(self) -> Tensor:
        """
        Lower-triangular matrix as an `[N, N]` `Tensor`.
        """
        return Tensor(self['l'])

    @property
    def u(self) -> Tensor:
        """
        Upper-triangular matrix as an `[N, N]` `Tensor`.
        """
        return Tensor(self['u'])


@post_op
def plu(x: Tensor) -> PLUFactorization:
    """Compute the PLU factorization of the given `matrix`.

    Args:
        `x`: a matrix with shape `[N, N]`

    Returns:
        A `[p, l, u]` list of `Tensor`s where
        `p`: a permutation matrix,
        `l`: lower triangular with unit diagonal elements,
        `u`: upper triangular.
    """

    def permute_rows(x: Tensor, p: Tensor, start_from: UInt) -> Map:

        @post_op
        def step(p: Tensor, x: Tensor, k: UInt) -> Map:
            p_k, p_kp1 = p[k].copy(), p[k + 1].copy()
            x_k, x_kp1 = x[k].copy(), x[k + 1].copy()

            return Map(After(
                [
                    p[k].write(p_kp1),
                    p[k + 1].write(p_k),
                    x[k].write(x_kp1),
                    x[k + 1].write(x_k),
                ],
                Map(p=p, x=x, k=k + 1)
            ))

        @post_op
        def cond(x: Tensor, k: UInt):
            return (k < UInt(Tuple(x.shape)[0]) - 1).logical_and(x[k, k].abs() < 1e-3)

        return Map(While(cond, step, Map(
            p=p.copy(),
            x=x.copy(),
            k=start_from,
        )))

    @post_op
    def step(p: Tensor, l: Tensor, u: Tensor, i: UInt) -> Map:
        pu = permute_rows(p=p, x=u, start_from=i)
        u = Tensor(pu['x'])
        p = Tensor(pu['p'])
        factor = Tensor(u[i + 1:, i] / u[i, i])
        return Map(After(
            when=[
                l[i + 1:, i].write(factor),
                u[i + 1:].write(u[i + 1:] - factor.expand_dims() * u[i]),
            ],
            then=Map(p=p, l=l, u=u, i=i + 1)))

    @post_op
    def cond(p: Tensor, l: Tensor, u: Tensor, i: UInt):
        return i < UInt(u.shape[0]) - 1

    return PLUFactorization(Map(While(cond, step, Map(
        p=identity(x.shape[0], F32).as_dense().copy(),
        l=identity(x.shape[0], F32).as_dense().copy(),
        u=x.copy(),
        i=UInt(0),
    ))))


def slogdet(x):
    """Compute the sign and log of the absolute value of the determinant of one or more square matrices."""

    # TODO: implement slogdet, cf. https://www.tensorflow.org/api_docs/python/tf/linalg/slogdet

    raise NotImplementedError(f"slogdet({x})")


def svd(matrix: Tensor) -> Tuple:
    """Return the singular value decomposition of the given `matrix`."""

    raise NotImplementedError("singular value decomposition")


def householder_bidiag(U: Tensor, W: Tensor, e: Tensor):
    """Householder's reduction to bidiagonal form"""
    @post_op
    def check_rows_num(cxt, U: Tensor, scale: F32, i: UInt, m: UInt, n: UInt, l: UInt):
        scale = F32(norm(U[i:, i]))**2

        # @post_op
        def change_rows(cxt, U: Tensor, scale: F32, i: UInt, l: UInt, n: UInt):
            s = F32(After(U[i:, i].write(U[i:, i] / scale),
                    F32(norm(U[i:, i]))))
            g = F32(If(F32(U[i, i]) >= F32(0), -s, s))*1
            h = After(U[i, i].write(F32(U[i, i]) - g), F32(F32(U[i, i]) * g - s**2))
            cxt.h = h

            @closure
            @post_op
            def step(j: UInt, U: Tensor) -> Map:
                return After(
                    U[i:, j].write(U[i:, j] + U[i:, i] *
                                   F32(F32(U[i:, i].mul(U[i:, j]).sum()) / cxt.h)),
                    Map(U=U, j=j + 1))

            @closure
            @post_op
            def cond(j: UInt):
                return j < n

            result = Map(While(cond, step, Map(
                U=U.copy(),
                j=l
            )))

            U = Tensor(result['U'])

            return After(U[i:, i].write(U[i:, i] * scale),Tuple([U, scale, g]))

        return If((i >= m).logical_or(scale <= EPS), Tuple([U, scale, 0.0]), change_rows(cxt, U=U, scale=scale, i=i, l=l, n=n))

    @post_op
    def check_cols_num(cxt, U:Tensor, scale: F32, i: UInt, m: UInt, n:UInt, l: UInt, e: Tensor):
        scale = F32(norm(U[i, l:]))**2
    
        # @post_op
        def change_cols(cxt, U: Tensor, scale: F32, i: UInt, l: UInt, n: UInt, e: Tensor):
            s = F32(After(U[i, l:].write(U[i, l:] / scale), F32(norm(U[i, l:]))))
            f = F32(U[i, l])
            g = F32(If(f >= F32(0), -s, s))*1
            h = F32(f * g - s)

            @closure
            @post_op
            def step(j: UInt, U: Tensor) -> Map:
                return After(
                    U[j, l:].write(U[j, l:] + e[l:] * F32(U[j, l:].mul(U[i, l:]).sum())),
                    Map(U=U, j=j + 1))
            @closure
            @post_op
            def cond(j: UInt):
                return j < m

            result = After(
                [U[i,l].write(f - g), e[l:].write(U[i,l:] / h)],
                Map(While(cond, step, Map(
                U=U.copy(),
                j=l
            ))))

            U = Tensor(result['U'])

            return After(U[i,l:].write(U[i,l:] * scale), Tuple([U, scale, g]))

        return If(((i >= m).logical_or(i == n-1)).logical_or(scale <= EPS), Tuple([U, scale, 0.0]), change_cols(cxt, U=U, scale=scale, i=i, l=l, n=n, e = e))
