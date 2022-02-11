from tinychain.collection.tensor import einsum, Dense, Schema, Sparse, Tensor
from tinychain.decorators import closure, get_op, post_op
from tinychain.ref import After, Get, If, MethodSubject, While
from tinychain.state import Map, Stream, Tuple
from tinychain.value import Bool, F64, UInt, F32, Int, Value, Number

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
    zero_diag = matrix - (matrix * eye)  # don't use eye.logical_not in case the matrix is sparse
    new_diag = eye * diag.expand_dims()
    return matrix.write(zero_diag + new_diag)


# TODO: vectorize to support a `Tensor` containing a batch of matrices
@post_op
def householder(cxt, x: Tensor) -> Tuple:
    """Compute the Householder vector of the given column vector `a`."""

    cxt.alpha = x[0]
    cxt.s = F32(If(Int(x.shape[0]) > 1, (x[1:]**2).sum(), 0.0))
    cxt.t = (cxt.alpha**2 + cxt.s)**0.5

    cxt.v = x.copy()
    cxt.v_zero = F64(If(cxt.alpha <= 0, cxt.alpha - cxt.t, -cxt.s / (cxt.alpha + cxt.t)))
    tau = If(cxt.s.abs() < EPS, 0, 2 * cxt.v_zero**2 / (cxt.s + cxt.v_zero ** 2))
    v = Tensor(If(Int(x.shape[0]) > 1, After(cxt.v[0].write(cxt.v_zero), cxt.v / cxt.v_zero), cxt.v))

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

    @closure(outer_cxt.m, cxt.householder)
    @post_op
    def qr_step(cxt, Q: Tensor, R: Tensor, k: UInt) -> Map:
        cxt.v, cxt.tau = Tuple(outer_cxt.householder(x=R[k:, k])).unpack(2)
        cxt.v_outer = Tensor(einsum("i,j->ij", [Tensor(cxt.v), Tensor(cxt.v)]))
        cxt.H = identity(outer_cxt.m, F64).as_dense().copy()
        cxt.H_sub = Tensor(cxt.H[k:, k:] - (Tensor(cxt.v_outer) * cxt.tau))
        return After(cxt.H[k:, k:].write(cxt.H_sub), {
            "Q": matmul(cxt.H, Q),
            "R": matmul(cxt.H, R),
        })

    state = Map(Q=identity(cxt.m, F64).as_dense(), R=x.copy())
    QR = Stream.range(cxt.n).fold("k", state, qr_step)
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

    @property
    def num_permutations(self) -> Tensor:
        """
        Upper-triangular matrix as an `[N, N]` `Tensor`.
        """
        return Tensor(self['num_permutations'])


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
            p_k, p_kp1 = p[start_from].copy(), p[k + 1].copy()
            x_k, x_kp1 = x[start_from].copy(), x[k + 1].copy()

            return Map(After(
                [
                    p[start_from].write(p_kp1),
                    p[k + 1].write(p_k),
                    x[start_from].write(x_kp1),
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
    def step(p: Tensor, l: Tensor, u: Tensor, i: UInt, num_permutations: UInt) -> Map:
        pu = permute_rows(p=p, x=u, start_from=i)
        u = Tensor(pu['x'])
        p = Tensor(pu['p'])
        n = UInt(pu['k']) - i
        factor = Tensor(u[i + 1:, i] / u[i, i])
        return Map(After(
            when=[
                l[i + 1:, i].write(factor),
                u[i + 1:].write(u[i + 1:] - factor.expand_dims() * u[i]),
            ],
            then=Map(p=p, l=l, u=u, i=i + 1, num_permutations=num_permutations + n)))

    @post_op
    def cond(p: Tensor, l: Tensor, u: Tensor, i: UInt):
        return i < UInt(u.shape[0]) - 1

    return PLUFactorization(Map(While(cond, step, Map(
        p=identity(x.shape[0], F32).as_dense().copy(),
        l=identity(x.shape[0], F32).as_dense().copy(),
        u=x.copy(),
        i=0,
        num_permutations=0,
        ))))


@post_op
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

    return diagonal(plu_result.u).product()*sign


@post_op
def slogdet(cxt, x: Dense) -> Tuple:
    """Compute the sign and log of the absolute value of the determinant of one or more square matrices.

    Args:
        `x`: a `Tensor` of square `matrix`es with shape `[N, M, M]`

    Returns:
        The `Tuple` of `Tensor`s `(sign, logdet)` where:
        `sign`: a `Tensor` of signs of determinants `{-1, +1}` with shape `[N,]`
        `logdet`: a `Tensor` of the natural log of the absolute values of determinants with shape `[N,]`
    """

    n = x.shape[0]
    cxt.sign_result = Dense.create([n])
    cxt.logdet_result = Dense.create([n])
    cxt.det = det

    @closure(x, cxt.det, cxt.sign_result, cxt.logdet_result)
    @get_op
    def step(i: UInt):
        d = cxt.det(x=x[i])
        logdet = F32(d.abs().log())
        sign = Int(If(d > 0, 1, -1))*1
        return [
            cxt.sign_result[i].write(sign),
            cxt.logdet_result[i].write(logdet),
        ]

    return After(Stream.range((0, n)).for_each(step), Tuple([cxt.sign_result, cxt.logdet_result]))


@post_op
def svd(cxt, A: Tensor, l=UInt(0), epsilon=F32(1e-5), max_iter=UInt(1000)) -> Tuple:
    cxt.shape = A.shape
    cxt.n_orig, cxt.m_orig = [UInt(dim) for dim in cxt.shape.unpack(2)]
    k = Int(If(l == UInt(0), Value.min(Int(cxt.n_orig), Int(cxt.m_orig)), l))
    A_orig = Tensor(A).copy()
    cxt.A1, n, m = Tuple(If(
        UInt(cxt.n_orig) > UInt(cxt.m_orig),
        Tuple([Tensor(matmul(Tensor(A).transpose(), A)), Number(Tensor(A).shape[1]), Number(Tensor(A).shape[1])]),
        Tuple(If(
            UInt(cxt.n_orig) < UInt(cxt.m_orig),
            Tuple([Tensor(matmul(A, Tensor(A).transpose())), Number(Tensor(A).shape[0]), Number(Tensor(A).shape[0])]),
            Tuple([A, cxt.n_orig, cxt.m_orig])
        )),
    )).unpack(3)

    Q = Dense.random_uniform([n, k]).abs()
    cxt.qr = qr
    Q, R = cxt.qr(x=Q).unpack(2)

    @closure(cxt.qr, cxt.A1)
    @post_op
    def step(i: UInt, Q_prev: Tensor, Q: Tensor, R: Tensor, err: F32):
        Z = Tensor(matmul(Tensor(cxt.A1), Tensor(Q)))
        _Q, _R = cxt.qr(x=Z).unpack(2)
        _err = F32(Tensor(_Q).sub(Q_prev).pow(2).sum())
        _Q_prev = Tensor(_Q).copy()
        return Map(i=i + 1, Q_prev=_Q_prev, Q=Tensor(_Q), R=Tensor(_R), err=_err)

    @closure(epsilon, max_iter)
    @post_op
    def cond(i: UInt, err: F32):
        return (F32(err).abs() > epsilon).logical_and(i < max_iter)

    result_loop = Map(While(cond, step, Map(
        i=UInt(0),
        Q_prev=Tensor(Q).copy(),
        Q=Tensor(Q).copy(),
        R=Tensor(R),
        err=F32(1.0)
        )))
    Q, R = result_loop['Q'], result_loop['R']

    singular_values = Tensor(Tensor(diagonal(R)).pow(0.5))
    cxt.eye = identity(singular_values.shape[0], F32).as_dense().copy()
    cxt.inv_matrix = (cxt.eye * singular_values.pow(-1))
    cxt.Q_T = Tensor(Q).transpose()
    cxt.vec_sing_values_upd = Map(If(
        cxt.n_orig == cxt.m_orig,
        Map(left_vecs=cxt.Q_T, right_vecs=cxt.Q_T, singular_values=(singular_values).pow(2)),
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
            cxt.vec_sing_values_upd
            ))

    return vec_sing_values['left_vecs'], vec_sing_values['singular_values'], vec_sing_values['right_vecs']
