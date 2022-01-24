from tinychain.collection.tensor import einsum, Dense, Schema, Sparse, Tensor
from tinychain.decorators import closure, get_op, post_op
from tinychain.ref import After, Get, If, MethodSubject, While
from tinychain.state import Map, Stream, Tuple
from tinychain.value import Bool, F64, Float, UInt, F32, Int, Number, String
from tinychain.error import BadRequest

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

    @closure(outer_cxt.m, cxt.householder)
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


def svd(matrix: Tensor) -> Tuple:
    """Return the singular value decomposition of the given `matrix`."""

    raise NotImplementedError("singular value decomposition")


def sign_like(a, b):
    return Number(If(b >= 0, a.abs(), -a.abs()))


def dot(x1, x2):
    return (x1 * x2).sum()


# TODO: make it work like numpy version
@post_op
def householder_vector(txn, U: Tensor, W: Tensor, e: Tensor) -> Map:
    """Householder's reduction to bidiagonal form"""

    txn.shape = U.shape
    txn.m, txn.n = [UInt(dim) for dim in txn.shape.unpack(2)]
    
    @closure(U, W, e, txn.m, txn.n)
    @post_op
    def step(cxt, i: UInt, scale: F32, g: F32) -> Map:

        @closure(U, txn.n)
        @post_op
        def update_cols(cxt, i: UInt, l: UInt, scale: F32) -> F32:
            cxt.col_i = U[i:, i]

            outer_cxt = cxt
            @closure(U, txn.n, i, outer_cxt.col_i)
            @post_op
            def step(cxt, j: UInt, h: F32) -> Map:
                cxt.col_j = U[i:, j]
                return After(
                    cxt.col_j.write(cxt.col_j + outer_cxt.col_i * dot(outer_cxt.col_i, cxt.col_j) / h),
                    Map(j=j + 1, h=h)
                )

            @closure(txn.n)
            @post_op
            def cond(j: UInt) -> Bool:
                return j < txn.n

            cxt.new_col = (cxt.col_i / scale).copy()
            cxt.descale_col = cxt.col_i.write(cxt.new_col)
            cxt.s = Float(After(cxt.descale_col, dot(cxt.col_i, cxt.col_i)))
            cxt.f = Float(After(cxt.descale_col, U[i, i]))
            cxt.g = -sign_like(cxt.s ** 0.5, cxt.f)

            cxt.h = Float(After(U[i, i].write(cxt.f - cxt.g), (cxt.f * cxt.g) - cxt.s))

            cxt.update_col = While(cond, step, Map(j=l, h=cxt.h))
            cxt.rescale_col = After(cxt.update_col, cxt.col_i.write(cxt.col_i * scale))
            return If(
                scale.abs() < EPS,
                BadRequest(String("scale of column {{i}} is {{scale}}").render(i=i, scale=scale)),
                After(cxt.rescale_col, cxt.g))

        @closure(U, e, txn.m, i)
        @post_op
        def update_rows(cxt, l: UInt, scale: F32) -> F32:
            cxt.row_i = U[i, l:]

            outer_cxt = cxt
            @closure(U, e, txn.m, i, l, outer_cxt.row_i)
            @post_op
            def step(cxt, j: UInt) -> Map:
                cxt.row_j = U[j, l:]
                cxt.update = cxt.row_j.write(cxt.row_j + e[l:] * dot(cxt.row_j, outer_cxt.row_i))
                return After(cxt.update, Map(j=j + 1))

            @closure(txn.m)
            @post_op
            def cond(j: UInt) -> Bool:
                return j < txn.m

            cxt.U_i_l = U[i, l]
            cxt.new_row = (cxt.row_i / scale).copy()
            cxt.descale = cxt.row_i.write(cxt.new_row)

            cxt.s_init = Float(After(cxt.descale, dot(cxt.row_i, cxt.row_i)))
            cxt.f = Float(After(cxt.descale, U[i, l]))
            cxt.g_init = -sign_like(cxt.s_init ** 0.5, cxt.f)

            cxt.h = After(cxt.U_i_l.write(cxt.f - cxt.g_init), (cxt.f * cxt.g_init) - cxt.s_init)
            cxt.s_final = After(e[l:].write(cxt.row_i / cxt.h), cxt.s_init)
            cxt.g_final = After(While(cond, step, Map(j=l)), cxt.g_init)

            cxt.rescale = After(cxt.s_final, cxt.row_i.write(cxt.row_i * scale))
            return After(cxt.rescale, cxt.g_final)

        cxt.update_cols = update_cols
        cxt.update_rows = update_rows

        cxt.update_e = e[i].write(scale * g)
        cxt.l = UInt(After(cxt.update_e, i + 1))
        cxt.scale_init = Float(If(i < txn.m, dot(U[i:, i], U[i:, i]), scale))
        cxt.g_init = Float(If(
            Bool(i < txn.m).logical_and(cxt.scale_init.abs() > EPS),
            cxt.update_cols(i=i, l=cxt.l, scale=cxt.scale_init),
            0.0))

        cxt.scale_new = Float(If(
            Bool(i < txn.m).logical_and(i != txn.n - 1),
            dot(U[i, cxt.l:], U[i, cxt.l:]),
            scale))

        cxt.scale_final = Float(After(W[i].write(cxt.scale_init * cxt.g_init), cxt.scale_new))

        g = If(
            Bool(i < txn.m).logical_and(i != txn.n - 1).logical_and(cxt.scale_init > EPS),
            cxt.update_rows(l=cxt.l, scale=cxt.scale_final),
            0.0)

        return Map(i=i + 1, scale=cxt.scale_final, g=g, U=U, W=W, e=e)

    @closure(txn.n)
    @post_op
    def cond(i: UInt) -> Bool:
        return i < txn.n

    return Map(While(cond, step, Map(i=0, scale=0.0, g=0.0, U=U, W=W, e=e)))


# TODO: make it work like numpy version
@post_op
def rht(txn, U: Tensor, V: Tensor, e: Tensor) -> Map:
    """Accumulation of right hand transformations (rht)"""

    txn.shape = U.shape
    txn.m, txn.n = [UInt(dim) for dim in txn.shape.unpack(2)]
    
    @closure(U, V, e, txn.m, txn.n)
    @post_op
    def step(cxt, i: UInt, l: UInt, g: F32) -> Map:
        
        @closure(U, V, e, txn.m, txn.n)
        @post_op
        def update_cols_wo_last(cxt, i: UInt, l: UInt, g: F32) -> F32:

            @closure(U, V, e, txn.m, txn.n)
            @post_op
            def update_cols(cxt, i: UInt, l: UInt, g: F32) -> F32:
                
                @closure(U, V, e, txn.m, txn.n)
                @post_op
                def step(cxt, j: UInt, l: UInt, i: UInt) -> Map:
                    cxt.col_j = V[l:, j]
                    cxt.update_col = cxt.col_j.write(cxt.col_j + V[l:, i] * dot(U[i, l:], cxt.col_j))
                    return Map(After(
                        cxt.update_col,
                        Map(j=j + 1, n=txn.n, l=l, i=i)
                    ))

                @closure(txn.n)
                @post_op
                def cond(j: UInt) -> Bool:
                    return j < txn.n

                loop = After(V[l:,i].write((U[i,l:] / U[i,l]) / g), While(cond, step, Map(j=l, n=txn.n, l=l, i=i)))

                return loop

            cxt.update_cols = update_cols
            cxt.res = cxt.update_cols(n=txn.n, l=l, i=i)
            return If((g != 0.0), cxt.res)

        cxt.update_g = Float(e[i].copy())
        cxt.update_cols_wo_last = update_cols_wo_last
        cxt.update_cols_funct = After(cxt.update_cols_wo_last(l=l, g=g), [V[i,l:].write(0.0), V[l:,i].write(0.0)])
        
        cxt.check_last_col = If(
            (i < txn.n-1),
            cxt.update_cols_funct)
        
        g_final = After([cxt.check_last_col, V[i,i].write(1.0)], cxt.update_g)

        return Map(i=i - 1, l=i, g=g_final, U=U, V=V, e=e)

    @post_op
    def cond(i: UInt) -> Bool:
        return i >= 0

    return Map(While(cond, step, Map(i=txn.n-1, l=0, g=0.0, U=U, V=V, e=e)))


# TODO: make it work like numpy version
@post_op
def lht(txn, U: Tensor, W: Tensor) -> Map:
    """Accumulation of left hand transformations (lht)"""

    txn.shape = U.shape
    txn.m, txn.n = [UInt(dim) for dim in txn.shape.unpack(2)]
    txn.min_m_n = UInt(If(txn.m < txn.n, txn.m, txn.n))
    
    @closure(U, W, txn.m, txn.n)
    @post_op
    def step(cxt, i: UInt, l: UInt, g: F32) -> Map:
        
        @closure(U, W, txn.m, txn.n)
        @post_op
        def update_cols(cxt, i: UInt, l: UInt, g: F32) -> F32:

            @closure(U, W, txn.m, txn.n)
            @post_op
            def step(j: UInt, l: UInt, i: UInt) -> Map:
                cxt.col_j = U[i:,j]
                cxt.update_col = cxt.col_j.write(cxt.col_j + U[i:, i] * ((dot(U[l:, i], U[l:, j])/U[i,i])*g))
                return Map(After(
                    cxt.update_col,
                    Map(j=j + 1, n=txn.n, l=l, i=i)
                ))

            @closure(txn.n)
            @post_op
            def cond(j: UInt) -> Bool:
                return j < txn.n

            cxt.update_g = 1.0 / g
            loop = After(cxt.update_g, While(cond, step, Map(j=l, n=txn.n, l=l, i=i)))
            update_U = After([loop, U[i:,i].write(U[i:,i]*cxt.update_g)], cxt.update_g)

            return If((g != 0.0), update_U, g)

        cxt.update_cols = update_cols
        cxt.new_l = UInt(i + 1)
        cxt.l = UInt(After(U[i, cxt.new_l:].write(0.0), cxt.new_l))
        cxt.new_g = Float(W[i])
        cxt.final_g = cxt.update_cols(i=i, l=cxt.l, g=cxt.new_g)


        return Map(i=i - 1, l=cxt.l, g=cxt.final_g, U=U, W=W)

    @post_op
    def cond(i: UInt) -> Bool:
        return i >= 0

    return Map(While(cond, step, Map(i=txn.min_m_n-1, l=0, g=0.0, U=U, W=W)))


@post_op
def golub_kahan(txn, U: Tensor, W: Tensor, e: Tensor, k: UInt, maxiter=30):
    """
    Diagonalization of the bidiagonal form: 
        - k is the kth singular value
        - loop over maxiter allowed iteration
    """

    raise NotImplementedError("golub kahan diagonalization")
