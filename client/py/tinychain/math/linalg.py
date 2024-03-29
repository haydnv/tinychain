import typing

from ..service import library_uri, Library
from ..collection.tensor import einsum, Dense, Sparse, Tensor
from ..decorators import closure, get as get_op, post
from ..error import BadRequest
from ..generic import Map, Tuple
from ..scalar.number import Number, UInt, F32, Int
from ..scalar.ref import after, cond, while_loop, Get
from ..scalar.value import Value, Version
from ..uri import URI

from .base import product
from .constants import NS

# from "Numerical Recipes in C" p. 65
EPS = F32(10**-6)


def diagonal(tensor):
    """Return the diagonal of the given `tensor` of matrices"""

    rtype = type(tensor) if isinstance(tensor, Tensor) else Tensor
    op = Get(URI(tensor, "diagonal"))
    return rtype(form=op)


def with_diagonal(matrix, diag):
    """Construct a view of the given `matrix` with the diagonal set to `diag`."""

    eye = Sparse.eye(matrix.shape[-2])
    return eye.cond(diag, matrix)


# TODO: replace this helper class with a `typing.TypedDict`
class PLUFactorization(Map):
    """PLU factorization of a given `[N, N]` matrix."""

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


# TODO: replace Tuple.range with a Stream after re-implementing Stream
class LinearAlgebra(Library):
    NAME = "linalg"
    VERSION = Version("0.0.0")

    __uri__ = library_uri(None, NS, NAME, VERSION)

    # TODO: vectorize to support a `Tensor` containing a batch of matrices
    @post
    def householder(self, cxt, x: Tensor) -> typing.Tuple[Tensor, Tensor]:
        """Compute the Householder vector of the given column vector `a`."""

        cxt.alpha = x[0]
        cxt.s = cond(x.shape[0] > 1, (x[1:]**2).sum(), 0.0)
        cxt.t = (cxt.alpha**2 + cxt.s)**0.5

        cxt.v = x.copy()
        cxt.v_zero = cond(cxt.alpha <= 0, cxt.alpha - cxt.t, -cxt.s / (cxt.alpha + cxt.t))
        tau = cond(cxt.s.abs() < EPS, 0, 2 * cxt.v_zero**2 / (cxt.s + cxt.v_zero**2))
        v = cond(Int(x.shape[0]) > 1, after(cxt.v[0].write(cxt.v_zero), cxt.v / cxt.v_zero), cxt.v)

        return v, tau

    # TODO: vectorize to support a `Tensor` containing a batch of matrices
    @post
    def qr(self, txn, a: Tensor) -> typing.Tuple[Tensor, Tensor]:
        """Compute the QR decomposition of the given matrix `a`"""

        txn.shape = a.shape
        txn.n, txn.m = txn.shape.unpack(2)

        txn.q_init = Dense.zeros([txn.n, txn.n])
        txn.u_init = Dense.zeros([txn.n, txn.n])

        txn.u = after(txn.u_init[:, 0].write(a[:, 0]), txn.u_init).copy()
        txn.q = after(
                after(txn.u, txn.q_init[:, 0].write(txn.u_init[:, 0] / norm(txn.u_init[:, 0]))),
                txn.q_init
            ).copy()

        @closure(a, txn.q, txn.u)
        @get_op
        def q_step(cxt, i: UInt) -> Tensor:
            @closure(a, i, txn.q, txn.u)
            @get_op
            def u_step(j: UInt) -> Map:
                col = txn.u[:, i].copy() - (txn.q[:, j] * (a[:, i] * txn.q[:, j]).sum())
                return txn.u[:, i].write(col)

            cxt.update_u = after(txn.u[:, i].write(a[:, i]), Tuple.range(i).for_each(u_step))
            cxt.update_q = after(
                cxt.update_u,
                txn.q[:, i].write(txn.u[:, i] / norm(txn.u[:, i])))

            return after(cxt.update_q, {})

        n = cond(txn.n <= txn.m, txn.n, txn.m)
        txn.update_q = Tuple.range((1, n)).for_each(q_step)
        txn._q = Tensor(after(txn.update_q, txn.q))
        txn._r = Dense.zeros([txn.n, txn.m])

        @closure(txn._r, a, txn._q, txn.m)
        @get_op
        def r_step(i: UInt):
            @closure(txn._r, a, txn._q, i, txn.m)
            @get_op
            def r_step_inner(j: UInt):
                return txn._r[i, j].write((a[:, j] * txn._q[:, i]).sum())

            return Tuple.range((i, txn.m)).for_each(r_step_inner)

        return after(Tuple.range(txn.n).for_each(r_step), (txn._q, txn._r))

    @post
    def plu(self, txn, x: Tensor) -> PLUFactorization:
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

                return after(
                    [
                        p[start_from].write(p_kp1),
                        p[k + 1].write(p_k),
                        x[start_from].write(x_kp1),
                        x[k + 1].write(x_k),
                    ],
                    {'p': p, 'x': x, 'k': k + 1}
                )

            @post
            def while_cond(cxt, x: Tensor, k: UInt):
                cxt.valid_k = k < (x.shape[0] - 1)
                cxt.valid_x_k_k = x[k, k].abs() < 1e-3
                return cxt.valid_k.logical_and(cxt.valid_x_k_k)

            return while_loop(while_cond, step, {
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
            return after(
                when=[
                    l[i + 1:, i].write(factor),
                    u[i + 1:].write(u[i + 1:] - factor.expand_dims() * u[i]),
                ],
                then=Map(p=p, l=l, u=u, i=i + 1, num_permutations=num_permutations + n))

        @post
        def factor_cond(u: Tensor, i: UInt):
            return i < UInt(u.shape[0]) - 1

        txn.factorization = while_loop(factor_cond, step, {
            'p': Sparse.eye(x.shape[0]).cast(F32).as_dense().copy(),
            'l': Sparse.eye(x.shape[0]).cast(F32).as_dense().copy(),
            'u': x.copy(),
            'i': 0,
            "num_permutations": 0,
        })

        return cond(
            _is_square(x),
            txn.factorization,
            BadRequest("PLU decomposition requires a square matrix, not {{x}}", x=x))

    @post
    def det(self, x: Tensor) -> F32:
        """Computes the determinant of a square matrix `x`.

        Args:
            `x`: a matrix with shape `[N, N]`

        Returns:
            The determinant for `x`
        """

        plu_result = self.plu(x=x)
        sign = Int(-1).pow(plu_result.num_permutations)
        determinant = diagonal(plu_result.u).product() * sign

        return cond(
            _is_square(x),
            determinant,
            BadRequest("determinant requires a square matrix, not {{x}}", x=x))

    @post
    def slogdet(self, cxt, x: Dense) -> typing.Tuple[Tensor, Tensor]:
        """
        Compute the sign and log of the absolute value of the determinant of one or more square matrices.

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

        cxt.copy = x.reshape(Tuple([cxt.batch_size]) + x.shape[-2:]).copy()

        @closure(cxt.copy, cxt.sign_result, cxt.logdet_result)
        @get_op
        def step(i: UInt):
            d = self.det(x=cxt.copy[i])
            logdet = F32(d.abs().log())
            sign = cond(d > 0, 1, -1) * 1
            return [
                cxt.sign_result[i].write(sign),
                cxt.logdet_result[i].write(logdet),
            ]

        sign, determinants = after(
            Tuple.range((0, cxt.batch_size)).for_each(step),
            [cxt.sign_result, cxt.logdet_result])

        return Tensor(sign).reshape(cxt.batch_shape), Tensor(determinants).reshape(cxt.batch_shape)

    @post
    def svd_matrix(self, cxt, A: Tensor, l=UInt(0), epsilon=EPS, max_iter=UInt(30)) -> typing.Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the singular value decomposition of the given matrix `A`

        Returns:
            `(U, s, V)`: :class:`Tensor` s such that `A` ~= `u * (identity([P, P]) * s) * v`, where `P = min(N, M)`.
        """

        cxt.shape = A.shape
        cxt.n_orig, cxt.m_orig = cxt.shape.unpack(2)
        k = cond(l == 0, Value.min(cxt.n_orig, cxt.m_orig), l)
        cxt.A_orig = A.copy()
        cxt.A1, n, m = cond(
            cxt.n_orig > cxt.m_orig,
            [A.transpose() @ A, Tensor(A).shape[1], Tensor(A).shape[1]],
            cond(
                cxt.n_orig < cxt.m_orig,
                [A @ Tensor(A).transpose(), A.shape[0], A.shape[0]],
                [A, cxt.n_orig, cxt.m_orig]
            ),
        ).unpack(3)  # TODO: this call to `unpack` should not be necessary

        Q, R = self.qr(a=Dense.random_uniform([n, k]).abs())

        @closure(cxt.A1)
        @post
        def step(i: UInt, Q_prev: Tensor, Q: Tensor):
            Z = Tensor(cxt.A1) @ Q
            _Q, _R = self.qr(a=Z)
            _err = _Q.sub(Q_prev).pow(2).sum()
            _Q_prev = _Q.copy()
            return Map(i=i + 1, Q_prev=_Q_prev, Q=_Q, R=_R, err=_err)

        cxt.step = step

        @closure(epsilon, max_iter)
        @post
        def while_cond(i: UInt, err: F32):
            return (F32(err).abs() > epsilon).logical_and(i < max_iter)

        cxt.cond = while_cond
        result_loop = while_loop(cxt.cond, cxt.step, Map(
            i=UInt(0),
            Q_prev=Tensor(Q).copy(),
            Q=Tensor(Q).copy(),
            R=Tensor(R),
            err=F32(1.0)))

        Q, R = Tensor(result_loop['Q']), Tensor(result_loop['R'])

        singular_values = diagonal(R).pow(0.5)
        cxt.eye = Sparse.eye(singular_values.shape[0]).as_dense().copy()
        cxt.inv_matrix = (cxt.eye * singular_values.pow(-1))
        cxt.Q_T = Q.transpose()

        cxt.vec_sing_values_upd = cond(
            cxt.n_orig == cxt.m_orig,
            Map(left_vecs=cxt.Q_T, right_vecs=cxt.Q_T, singular_values=singular_values.pow(2)),
            Map(
                left_vecs=einsum('ij,jk->ik', [einsum('ij,jk->ik', [cxt.A_orig, Q]), cxt.inv_matrix]),
                right_vecs=cxt.Q_T,
                singular_values=singular_values))

        vec_sing_values = cond(
            cxt.n_orig < cxt.m_orig,
            Map(
                left_vecs=cxt.Q_T,
                right_vecs=einsum('ij,jk->ik', [einsum('ij,jk->ik', [cxt.inv_matrix, Q]), cxt.A_orig]),
                singular_values=singular_values),
            cxt.vec_sing_values_upd)

        return vec_sing_values['left_vecs'], vec_sing_values['singular_values'], vec_sing_values['right_vecs']

    # TODO: update to support `Tensor` (not just `Dense`) after `Sparse.concatenate` is implemented
    @post
    def svd_parallel(self, txn, A: Tensor, l=UInt(0), epsilon=EPS, max_iter=UInt(30)) -> typing.Tuple[Tensor, Tensor, Tensor]:
        """
        Given a `Tensor` of `matrices`, return the singular value decomposition `(s, u, v)` of each matrix.

        Currently only implemented for `Dense` matrices.
        """

        txn.N, txn.M = A.shape[-2:].unpack(2)
        txn.batch_shape = A.shape[:-2]
        txn.num_matrices = product(txn.batch_shape)

        txn.matrices = A.reshape([txn.num_matrices, txn.N, txn.M]).copy()

        @closure(txn.matrices, l, epsilon, max_iter)
        @get_op
        def matrix_svd(i: UInt) -> typing.Tuple[Tensor, Tensor, Tensor]:
            return self.svd_matrix(A=txn.matrices[i], l=l, epsilon=epsilon, max_iter=max_iter)

        txn.indices = Tuple.range(txn.num_matrices)
        txn.UsV_tuples = txn.indices.map(matrix_svd)

        def getter(j):
            @closure(txn.UsV_tuples)
            @get_op
            def getter(i: UInt) -> Dense:
                tensor = Tensor(Tuple(txn.UsV_tuples[i])[j])
                return tensor.expand_dims(0)

            return getter

        txn.U = Dense.concatenate(txn.indices.map(getter(0)), axis=0)
        txn.s = Dense.concatenate(txn.indices.map(getter(1)), axis=0)
        txn.V = Dense.concatenate(txn.indices.map(getter(2)), axis=0)

        return (
            txn.U.reshape(Tuple.concatenate(txn.batch_shape, txn.U.shape[1:])).copy(),
            txn.s.reshape(Tuple.concatenate(txn.batch_shape, [Number.min(txn.N, txn.M)])).copy(),
            txn.V.reshape(Tuple.concatenate(txn.batch_shape, txn.V.shape[1:])).copy(),
        )

    @post
    def svd(self, A: Tensor, l=UInt(0), epsilon=EPS, max_iter=UInt(30)) -> typing.Tuple[Tensor, Tensor, Tensor]:
        """
        Computes `svd_matrix` for each matrix in `A`.

        For `A` with shape `[..., N, M]`, `svd` returns a tuple `(U, s, V)` such that
         `A[...]` ~= `u[...] * (identity([P, P]) * s[...]) * v[...]`, where `P = min(N, M)`.
        """

        return cond(
            A.ndim == 2,
            self.svd_matrix(A=A, l=l, epsilon=epsilon, max_iter=max_iter),
            self.svd_parallel(A=A, l=l, epsilon=epsilon, max_iter=max_iter))


def norm(x):
    """Helper function to compute the norm of a matrix `x`"""

    return (x**2).sum()**0.5


def _is_square(x):
    return (x.ndim == 2).logical_and(x.shape[0] == x.shape[1])
