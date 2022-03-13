import typing

from ..collection.tensor import einsum, Dense, Sparse, Tensor
from ..decorators import closure, get as get_op, post
from ..error import BadRequest
from ..generic import Map, Tuple
from ..scalar.number import Number, Bool, F64, UInt, F32, Int
from ..scalar.ref import After, Get, If, MethodSubject, While
from ..scalar.value import Value
from ..state import Stream

from .base import product

# from "Numerical Recipes in C" p. 65
EPS = 10**-6


def diagonal(matrix):
    """Return the diagonal of the given `matrix`"""

    rtype = type(matrix) if isinstance(matrix, Tensor) else Tensor
    op = Get(MethodSubject(matrix, "diagonal"))
    return rtype(op)


def identity(size, dtype=Bool):
    """Return an identity matrix with dimensions `[size, size]`."""

    schema = ([size, size], dtype)
    elements = Stream.range((0, size)).map(get_op(lambda i: ((i, i), 1)))
    return Sparse.copy_from(schema, elements)


def set_diagonal(matrix, diag):
    """Set the diagonal of the given `matrix` to `diag`."""

    eye = identity(matrix.shape[0])
    zero_diag = matrix - (matrix * eye)  # don't use eye.logical_not in case the matrix is sparse
    new_diag = eye * diag.expand_dims()
    return matrix.write(zero_diag + new_diag)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
@post
def householder(cxt, x: Tensor) -> Tuple:
    """Compute the Householder vector of the given column vector `a`."""

    cxt.alpha = x[0]
    cxt.s = F32(If(Int(x.shape[0]) > 1, (x[1:]**2).sum(), 0.0))
    cxt.t = (cxt.alpha**2 + cxt.s)**0.5

    cxt.v = x.copy()
    cxt.v_zero = F64(If(cxt.alpha <= 0, cxt.alpha - cxt.t, -cxt.s / (cxt.alpha + cxt.t)))
    tau = If(cxt.s.abs() < EPS, 0, 2 * cxt.v_zero**2 / (cxt.s + cxt.v_zero**2))
    v = Tensor(If(Int(x.shape[0]) > 1, After(cxt.v[0].write(cxt.v_zero), cxt.v / cxt.v_zero), cxt.v))

    return v, tau


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
@post
def qr(cxt, a: Tensor) -> typing.Tuple[Tensor, Tensor]:
    """Compute the QR decomposition of the given matrix `a`"""

    cxt.shape = a.shape
    cxt.n, cxt.m = cxt.shape.unpack(2)

    cxt.q_init = Dense.zeros([cxt.n, cxt.n])
    cxt.u_init = Dense.zeros([cxt.n, cxt.n])

    cxt.u = Tensor(After(cxt.u_init[:, 0].write(a[:, 0]), cxt.u_init)).copy()
    cxt.q = Tensor(
        After(
            After(cxt.u, cxt.q_init[:, 0].write(cxt.u_init[:, 0] / norm(cxt.u_init[:, 0]))),
            cxt.q_init
        )).copy()

    @closure(a)
    @post
    def q_step(cxt, q: Tensor, u: Tensor, i: UInt) -> Map:
        @closure(a)
        @post
        def u_step(q: Tensor, u: Tensor, i: UInt, j: UInt) -> Map:
            return After(u[:, i].write(u[:, i].copy() - q[:, j].mul(a[:, i].mul(q[:, j]).sum())), Map(q=q, u=u, i=i))

        state_u_step = Map(q=q, u=Tensor(After(u[:, i].write(a[:, i]), u)), i=i)
        cxt.update_u = Stream.range(i).fold('j', state_u_step, u_step)

        return After(
            cxt.update_u,
            Map(q=Tensor(After(q[:, i].write(u[:, i] / F32(norm(u[:, i]))), q)), u=Tensor(cxt.update_u['u'])))

    state_q_step = Map(q=cxt.q, u=cxt.u)
    cxt.update_q = Stream.range((1, UInt(cxt.n))).fold('i', state_q_step, q_step)
    cxt._q = Tensor(After(cxt.update_q, cxt.update_q['q']))
    cxt._r = Tensor(After(cxt._q, Dense.zeros([cxt.n, cxt.m])))

    @closure(cxt._r, a, cxt._q, cxt.m)
    @get_op
    def r_step(i: UInt):
        @closure(cxt._r, a, cxt._q, i, cxt.m)
        @get_op
        def r_step_inner(j: UInt):
            return cxt._r[i, j].write(a[:, j].mul(cxt._q[:, i]).sum())

        return Stream.range((i, cxt.m)).for_each(r_step_inner)

    update_r = Stream.range(cxt.n).for_each(r_step)
    qr_factorization = Map(After(update_r, Map(q=cxt._q, r=cxt._r)))

    return qr_factorization['q'], qr_factorization['r']


# TODO: replace this helper class with a `typing.TypedDict`
class PLUFactorization(Map):
    """
    PLU factorization of a given `[N, N]` matrix.
    """

    @property
    def p(self) -> Tensor:
        """Permutation matrix as an `[N, N]` `Tensor`"""

        return Tensor(self['p'])

    @property
    def l(self) -> Tensor:
        """Lower-triangular matrix as an `[N, N]` `Tensor`"""

        return Tensor(self['l'])

    @property
    def u(self) -> Tensor:
        """Upper-triangular matrix as an `[N, N]` `Tensor`"""

        return Tensor(self['u'])

    @property
    def num_permutations(self) -> UInt:
        """The number of permutations calculated during the factorization"""

        return UInt(self['num_permutations'])


@post
def plu(txn, x: Tensor) -> PLUFactorization:
    """Compute the PLU factorization of the given matrix `x`.

    Args:
        `x`: a matrix with shape `[N, N]`

    Returns `(p, l, u)` where
        `p` is the permutation matrix,
        `l` is lower triangular with unit diagonal elements, and
        `u` is upper triangular.
    """

    # TODO: use a TypedDict as the return annotation
    @post
    def permute_rows(x: Tensor, p: Tensor, start_from: UInt) -> Map:
        @closure(start_from)
        @post
        def step(p: Tensor, x: Tensor, k: UInt) -> Map:
            p_k, p_kp1 = p[start_from].copy(), p[k + 1].copy()
            x_k, x_kp1 = x[start_from].copy(), x[k + 1].copy()

            return After(
                [
                    p[start_from].write(p_kp1),
                    p[k + 1].write(p_k),
                    x[start_from].write(x_kp1),
                    x[k + 1].write(x_k),
                ],
                {'p': p, 'x': x, 'k': k + 1}
            )

        @post
        def cond(cxt, x: Tensor, k: UInt):
            cxt.valid_k = k < (x.shape[0] - 1)
            cxt.valid_x_k_k = x[k, k].abs() < 1e-3
            return cxt.valid_k.logical_and(cxt.valid_x_k_k)

        return While(cond, step, {
            'p': p.copy(),
            'x': x.copy(),
            'k': start_from
        })

    txn.permute_rows = permute_rows

    @closure(txn.permute_rows)
    @post
    def step(p: Tensor, l: Tensor, u: Tensor, i: UInt, num_permutations: UInt) -> Map:
        pu = txn.permute_rows(p=p, x=u, start_from=i)
        u = Tensor(pu['x'])
        p = Tensor(pu['p'])
        n = UInt(pu['k']) - i
        factor = Tensor(u[i + 1:, i] / u[i, i])
        return After(
            when=[
                l[i + 1:, i].write(factor),
                u[i + 1:].write(u[i + 1:] - factor.expand_dims() * u[i]),
            ],
            then=Map(p=p, l=l, u=u, i=i + 1, num_permutations=num_permutations + n))

    @post
    def cond(u: Tensor, i: UInt):
        return i < UInt(u.shape[0]) - 1

    txn.factorization = While(cond, step, {
        'p': identity(x.shape[0], F32).as_dense().copy(),
        'l': identity(x.shape[0], F32).as_dense().copy(),
        'u': x.copy(),
        'i': 0,
        "num_permutations": 0,
    })

    return If(
        _is_square(x),
        txn.factorization,
        BadRequest("PLU decomposition requires a square matrix, not {{x}}", x=x))


@post
def det(cxt, x: Tensor) -> F32:
    """Computes the determinant of square `matrix`.

    Args:
        `x`: a matrix with shape `[N, N]`

    Returns:
        The determinant for `x`
    """

    cxt.plu = plu
    plu_result = cxt.plu(x=x)
    sign = Int(-1).pow(plu_result.num_permutations)
    determinant = diagonal(plu_result.u).product() * sign

    return If(
        _is_square(x),
        determinant,
        BadRequest("determinant requires a square matrix, not {{x}}", x=x))


@post
def slogdet(cxt, x: Dense) -> typing.Tuple[Tensor, Tensor]:
    """Compute the sign and log of the absolute value of the determinant of one or more square matrices.

    Args:
        `x`: a `Tensor` of square matrices with shape `[..., M, M]`

    Returns:
        `(sign, logdet)` where:
            `sign` is a `Tensor` of signs of determinants `{-1, +1}` with shape `[...]`
            `logdet` is a `Tensor` of the natural log of the absolute values of determinants with shape `[...]`
    """

    cxt.batch_shape = x.shape[:-2]
    cxt.batch_size = product(cxt.batch_shape)

    cxt.sign_result = Dense.create([cxt.batch_size])
    cxt.logdet_result = Dense.create([cxt.batch_size])
    cxt.det = det

    cxt.copy = x.reshape(Tuple([cxt.batch_size]) + x.shape[-2:]).copy()

    @closure(cxt.copy, cxt.det, cxt.sign_result, cxt.logdet_result)
    @get_op
    def step(i: UInt):
        d = cxt.det(x=cxt.copy[i])
        logdet = F32(d.abs().log())
        sign = Int(If(d > 0, 1, -1)) * 1
        return [
            cxt.sign_result[i].write(sign),
            cxt.logdet_result[i].write(logdet),
        ]

    result = After(Stream.range((0, cxt.batch_size)).for_each(step), [cxt.sign_result, cxt.logdet_result])
    sign, determinants = Tuple(result).unpack(2)

    return Tensor.reshape(sign, cxt.batch_shape), Tensor.reshape(determinants, cxt.batch_shape)


@post
def svd_matrix(cxt, A: Tensor, l=UInt(0), epsilon=F32(1e-5), max_iter=UInt(30)) -> typing.Tuple[Tensor, Tensor, Tensor]:
    """
    Compute the singular value decomposition of the given matrix `A`

    Returns:
        `(U, s, V)`: :class:`Tensor` s such that `A` ~= `u * (identity([P, P]) * s) * v`, where `P = min(N, M)`.
    """

    cxt.qr = qr

    cxt.shape = A.shape
    cxt.n_orig, cxt.m_orig = cxt.shape.unpack(2)
    k = Int(If(l == UInt(0), Value.min(cxt.n_orig, cxt.m_orig), l))
    A_orig = A.copy()
    cxt.A1, n, m = Tuple(If(
        UInt(cxt.n_orig) > UInt(cxt.m_orig),
        [A.transpose() @ A, Tensor(A).shape[1], Tensor(A).shape[1]],
        If(
            cxt.n_orig < cxt.m_orig,
            [A @ Tensor(A).transpose(), A.shape[0], A.shape[0]],
            [A, cxt.n_orig, cxt.m_orig]
        ),
    )).unpack(3)

    Q, R = cxt.qr(a=Dense.random_uniform([n, k]).abs())

    @closure(cxt.qr, cxt.A1)
    @post
    def step(i: UInt, Q_prev: Tensor, Q: Tensor):
        Z = Tensor(cxt.A1) @ Q
        _Q, _R = cxt.qr(a=Z)
        _err = _Q.sub(Q_prev).pow(2).sum()
        _Q_prev = _Q.copy()
        return Map(i=i + 1, Q_prev=_Q_prev, Q=_Q, R=_R, err=_err)

    cxt.step = step

    @closure(epsilon, max_iter)
    @post
    def cond(i: UInt, err: F32):
        return (F32(err).abs() > epsilon).logical_and(i < max_iter)

    cxt.cond = cond
    result_loop = Map(While(cxt.cond, cxt.step, Map(
        i=UInt(0),
        Q_prev=Tensor(Q).copy(),
        Q=Tensor(Q).copy(),
        R=Tensor(R),
        err=F32(1.0))))

    Q, R = Tensor(result_loop['Q']), Tensor(result_loop['R'])

    singular_values = diagonal(R).pow(0.5)
    cxt.eye = identity(singular_values.shape[0], F32).as_dense().copy()
    cxt.inv_matrix = (cxt.eye * singular_values.pow(-1))
    cxt.Q_T = Q.transpose()

    cxt.vec_sing_values_upd = Map(If(
        cxt.n_orig == cxt.m_orig,
        Map(left_vecs=cxt.Q_T, right_vecs=cxt.Q_T, singular_values=singular_values.pow(2)),
        Map(
            left_vecs=einsum('ij,jk->ik', [einsum('ij,jk->ik', [A_orig, Q]), cxt.inv_matrix]),
            right_vecs=cxt.Q_T,
            singular_values=singular_values)))

    vec_sing_values = Map(If(
        cxt.n_orig < cxt.m_orig,
        Map(
            left_vecs=cxt.Q_T,
            right_vecs=einsum('ij,jk->ik', [einsum('ij,jk->ik', [cxt.inv_matrix, Q]), A_orig]),
            singular_values=singular_values),
        cxt.vec_sing_values_upd))

    return vec_sing_values['left_vecs'], vec_sing_values['singular_values'], vec_sing_values['right_vecs']


# TODO: update to support `Tensor` (not just `Dense`) after `Sparse.concatenate` is implemented
@post
def svd_parallel(txn, A: Tensor, l=UInt(0), epsilon=F32(1e-5), max_iter=UInt(30)) -> typing.Tuple[Tensor, Tensor, Tensor]:
    """
    Given a `Tensor` of `matrices`, return the singular value decomposition `(s, u, v)` of each matrix.

    Currently only implemented for `Dense` matrices.
    """

    txn.svd = svd_matrix

    txn.N, txn.M = A.shape[-2:].unpack(2)
    txn.batch_shape = A.shape[:-2]
    txn.num_matrices = product(txn.batch_shape)

    txn.matrices = A.reshape([txn.num_matrices, txn.N, txn.M]).copy()

    @closure(txn.svd, txn.matrices, l, epsilon, max_iter)
    @get_op
    def matrix_svd(i: UInt) -> typing.Tuple[Tensor, Tensor, Tensor]:
        return txn.svd(A=txn.matrices[i], l=l, epsilon=epsilon, max_iter=max_iter)

    # TODO: replace Tuple.range with Stream.range after updating Tensor.concatenate to accept a Stream of Tensors
    txn.indices = Tuple.range(txn.num_matrices)
    txn.UsV_tuples = txn.indices.map(matrix_svd)

    def getter(j):
        @closure(txn.UsV_tuples)
        @get_op
        def getter(i: UInt) -> Dense:
            return Tensor.expand_dims(Tuple(txn.UsV_tuples[i])[j], 0)

        return getter

    txn.U = Dense.concatenate(txn.indices.map(getter(0)), axis=0)
    txn.s = Dense.concatenate(txn.indices.map(getter(1)), axis=0)
    txn.V = Dense.concatenate(txn.indices.map(getter(2)), axis=0)

    return (
        txn.U.reshape(Tuple.concatenate(txn.batch_shape, txn.U.shape[1:])),
        txn.s.reshape(Tuple.concatenate(txn.batch_shape, [Number.min(txn.N, txn.M)])),
        txn.V.reshape(Tuple.concatenate(txn.batch_shape, txn.V.shape[1:]))
    )


@post
def svd(cxt, A: Tensor, l=UInt(0), epsilon=F32(1e-5), max_iter=UInt(30)) -> typing.Tuple[Tensor, Tensor, Tensor]:
    """
    Computes `svd_matrix` for each matrix in `A`.

    For `A` with shape `[..., N, M]`, `svd` returns a tuple `(U, s, V)` such that
     `A[...]` ~= `u[...] * (identity([P, P]) * s[...]) * v[...]`, where `P = min(N, M)`.
    """

    cxt.svd_matrix = svd_matrix
    cxt.svd_parallel = svd_parallel

    return If(
        A.ndim == 2,
        cxt.svd_matrix(A=A, l=l, epsilon=epsilon, max_iter=max_iter),
        cxt.svd_parallel(A=A, l=l, epsilon=epsilon, max_iter=max_iter))


def _is_square(x):
    return (x.ndim == 2).logical_and(x.shape[0] == x.shape[1])
