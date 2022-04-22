/// A [`Tensor`], an n-dimensional array of [`Number`]s which supports basic math and logic
use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::TryFutureExt;
use futures::stream::{StreamExt, TryStreamExt};
use itertools::Itertools;
use log::debug;
use safecast::*;

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::{ComplexType, FloatType, IntType, Number, NumberType, UIntType, Value, ValueType};
use tcgeneric::{
    label, path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCBoxTryFuture,
    TCPathBuf, Tuple,
};

use stream::ReadValueAt;

pub use afarray::{print_af_info, Array};
pub use bounds::{AxisBounds, Bounds, Shape};
pub use dense::{arg_sort, BlockListFile, DenseAccess, DenseAccessor, DenseTensor, DenseWrite};
pub use sparse::{SparseAccess, SparseAccessor, SparseTable, SparseTensor, SparseWrite};

mod bounds;
mod dense;
mod sparse;
mod stream;
mod transform;

const ERR_COMPLEX_EXPONENT: &str = "raising to a complex power is not supported";
const ERR_INF: &str = "Tensor combination resulted in an infinite value";
const ERR_NAN: &str = "Tensor combination resulted in a non-numeric value";

const PREFIX: PathLabel = path_label(&["state", "collection", "tensor"]);

/// The file extension of a [`Tensor`]
pub const EXT: &str = "array";

/// The schema of a [`Tensor`]
#[derive(Clone)]
pub struct Schema {
    pub shape: Shape,
    pub dtype: NumberType,
}

impl Schema {
    /// Raise an error if this schema is invalid.
    ///
    /// `debug_info` should be the name of the calling Op, as a user would understand it.
    pub fn validate(&self, debug_info: &'static str) -> TCResult<()> {
        self.shape.validate(debug_info)?;

        fn err_nonspecific(dtype: NumberType) -> TCResult<()> {
            Err(TCError::bad_request(
                "tensor requires a specific Number type, not",
                dtype,
            ))
        }

        match self.dtype {
            NumberType::Complex(ComplexType::Complex) => err_nonspecific(self.dtype),
            NumberType::Float(FloatType::Float) => err_nonspecific(self.dtype),
            NumberType::Int(IntType::Int) => err_nonspecific(self.dtype),
            NumberType::UInt(UIntType::UInt) => err_nonspecific(self.dtype),
            _ => Ok(()),
        }
    }
}

impl TryCastFrom<Value> for Schema {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Tuple(tuple) => TryCastInto::<(Vec<u64>, TCPathBuf)>::can_cast_into(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(tuple) => {
                let (shape, dtype): (Vec<u64>, TCPathBuf) = tuple.opt_cast_into()?;
                let shape = Shape::from(shape);
                let dtype = ValueType::from_path(&dtype)?;
                match dtype {
                    ValueType::Number(dtype) => Some(Schema { shape, dtype }),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

impl CastFrom<Schema> for Value {
    fn cast_from(schema: Schema) -> Self {
        let Schema { shape, dtype } = schema;
        let shape = shape.into_vec();
        let shape = shape.into_iter().map(Value::from).collect::<Tuple<Value>>();
        let dtype = ValueType::from(dtype).path();
        (shape, dtype).cast_into()
    }
}

#[async_trait]
impl de::FromStream for Schema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        debug!("deserialize Tensor schema");
        let schema = Value::from_stream(cxt, decoder).await?;
        Self::try_cast_from(schema, |v| de::Error::invalid_value(v, "a Tensor schema"))
    }
}

#[async_trait]
impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let Schema { shape, dtype } = self;
        (shape.to_vec(), ValueType::from(dtype).path()).into_stream(encoder)
    }
}

impl fmt::Display for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "shape {}, dtype {}", self.shape, self.dtype)
    }
}

impl From<(Shape, NumberType)> for Schema {
    fn from(schema: (Shape, NumberType)) -> Self {
        let (shape, dtype) = schema;
        Self { shape, dtype }
    }
}

/// The address of an individual element in a [`Tensor`].
pub type Coord = Vec<u64>;

/// Basic properties common to all [`Tensor`]s
pub trait TensorAccess {
    /// The datatype of this [`Tensor`]
    fn dtype(&self) -> NumberType;

    /// The number of dimensions of this [`Tensor`]
    fn ndim(&self) -> usize;

    /// The shape of this [`Tensor`]
    fn shape(&'_ self) -> &'_ Shape;

    /// The number of elements in this [`Tensor`]
    fn size(&self) -> u64;
}

/// A [`Tensor`] instance
pub trait TensorInstance {
    /// A dense representation of this [`Tensor`]
    type Dense: TensorInstance;

    /// A sparse representation of this [`Tensor`]
    type Sparse: TensorInstance;

    /// Return a dense representation of this [`Tensor`].
    fn into_dense(self) -> Self::Dense;

    /// Return a sparse representation of this [`Tensor`].
    fn into_sparse(self) -> Self::Sparse;
}

/// [`Tensor`] boolean operations.
pub trait TensorBoolean<O> {
    /// The result type of a boolean operation.
    type Combine: TensorInstance;

    /// The result type of a boolean operation which may ignore right-hand values.
    type LeftCombine: TensorInstance;

    /// Logical and
    fn and(self, other: O) -> TCResult<Self::LeftCombine>;

    /// Logical or
    fn or(self, other: O) -> TCResult<Self::Combine>;

    /// Logical xor
    fn xor(self, other: O) -> TCResult<Self::Combine>;
}

/// [`Tensor`] boolean operations in relation to a constant.
pub trait TensorBooleanConst {
    /// The return type of a boolean operation.
    type Combine: TensorInstance;
    /// The return type of a boolean operation with a result expected to be dense.
    type DenseCombine: TensorInstance;

    /// Logical and
    fn and_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Logical or
    fn or_const(self, other: Number) -> TCResult<Self::DenseCombine>;

    /// Logical xor
    fn xor_const(self, other: Number) -> TCResult<Self::DenseCombine>;
}

/// Tensor comparison operations
pub trait TensorCompare<O> {
    /// The result of a comparison operation
    type Compare: TensorInstance;

    /// The result of a comparison operation which can only return a dense [`Tensor`]
    type Dense: TensorInstance;

    /// Element-wise equality
    fn eq(self, other: O) -> TCResult<Self::Dense>;

    /// Element-wise greater-than
    fn gt(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise greater-or-equal
    fn gte(self, other: O) -> TCResult<Self::Dense>;

    /// Element-wise less-than
    fn lt(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise less-or-equal
    fn lte(self, other: O) -> TCResult<Self::Dense>;

    /// Element-wise not-equal
    fn ne(self, other: O) -> TCResult<Self::Compare>;
}

/// Tensor-constant comparison operations
pub trait TensorCompareConst {
    /// The result of a comparison operation
    type Compare: TensorInstance;

    /// Element-wise equality
    fn eq_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise greater-than
    fn gt_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise greater-or-equal
    fn gte_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise less-than
    fn lt_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise less-or-equal
    fn lte_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise not-equal
    fn ne_const(self, other: Number) -> TCResult<Self::Compare>;
}

/// [`Tensor`] linear algebra operations
#[async_trait]
pub trait TensorDiagonal<D: Dir> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<D>;

    /// The type of [`Tensor`] returned by `diagonal`
    type Diagonal: TensorAccess;

    async fn diagonal(self, txn: Self::Txn) -> TCResult<Self::Diagonal>;
}

/// [`Tensor`] I/O operations
#[async_trait]
pub trait TensorIO<D: Dir> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<D>;

    /// Read a single value from this [`Tensor`].
    async fn read_value(self, txn: Self::Txn, coord: Coord) -> TCResult<Number>;

    /// Write a single value to the slice of this [`Tensor`] with the given [`Bounds`].
    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()>;

    /// Overwrite a single element of this [`Tensor`].
    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

/// [`Tensor`] I/O operations which accept another [`Tensor`] as an argument
#[async_trait]
pub trait TensorDualIO<D: Dir, O> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<D>;

    /// Overwrite the slice of this [`Tensor`] given by [`Bounds`] with the given `value`.
    async fn write(self, txn: Self::Txn, bounds: Bounds, value: O) -> TCResult<()>;
}

/// [`Tensor`] indexing operations
#[async_trait]
pub trait TensorIndex<D: Dir> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<D>;

    /// The type of [`Tensor`] returned by `argmax`.
    type Index: TensorInstance;

    /// Return the indices of the maximum values in this [`Tensor`] along the given `axis`.
    async fn argmax(self, txn: Self::Txn, axis: usize) -> TCResult<Self::Index>;

    /// Return the offset of the maximum value in this [`Tensor`].
    async fn argmax_all(self, txn: Self::Txn) -> TCResult<u64>;
}

/// [`Tensor`] math operations
pub trait TensorMath<D: Dir, O> {
    /// The result type of a math operation
    type Combine: TensorInstance;

    /// The result type of a math operation which may ignore right-hand-side values
    type LeftCombine: TensorInstance;

    /// Add two tensors together.
    fn add(self, other: O) -> TCResult<Self::Combine>;

    /// Divide `self` by `other`.
    fn div(self, other: O) -> TCResult<Self::LeftCombine>;

    /// Element-wise logarithm of `self` with respect to the given `base`.
    fn log(self, base: O) -> TCResult<Self::LeftCombine>;

    /// Multiply two tensors together.
    fn mul(self, other: O) -> TCResult<Self::LeftCombine>;

    /// Raise `self` to the power of `other`.
    fn pow(self, other: O) -> TCResult<Self::LeftCombine>;

    /// Subtract `other` from `self`.
    fn sub(self, other: O) -> TCResult<Self::Combine>;
}

/// [`Tensor`] constant math operations
pub trait TensorMathConst {
    /// The return type of a math operation
    type Combine: TensorInstance;
    /// The return type of a math operation with a result expected to be dense
    type DenseCombine: TensorInstance;

    /// Add a constant to this tensor
    fn add_const(self, other: Number) -> TCResult<Self::DenseCombine>;

    /// Divide `self` by `other`.
    fn div_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Element-wise logarithm
    fn log_const(self, base: Number) -> TCResult<Self::Combine>;

    /// Multiply `self` by `other`.
    fn mul_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Raise `self` to the power `other`.
    fn pow_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Subtract `other` from `self`.
    fn sub_const(self, other: Number) -> TCResult<Self::DenseCombine>;
}

/// Methods to access this [`Tensor`] as a persistent type.
pub trait TensorPersist: Sized {
    type Persistent;

    fn as_persistent(self) -> Option<Self::Persistent> {
        None
    }
}

/// [`Tensor`] reduction operations
pub trait TensorReduce<D: Dir> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<D>;

    /// The result type of a reduce operation
    type Reduce: TensorInstance;

    /// Return the product of this [`Tensor`] along the given `axis`.
    fn product(self, axis: usize) -> TCResult<Self::Reduce>;

    /// Return the product of all elements in this [`Tensor`].
    fn product_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number>;

    /// Return the sum of this [`Tensor`] along the given `axis`.
    fn sum(self, axis: usize) -> TCResult<Self::Reduce>;

    /// Return the sum of all elements in this [`Tensor`].
    fn sum_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number>;
}

/// [`Tensor`] transforms
pub trait TensorTransform {
    /// A broadcast [`Tensor`]
    type Broadcast: TensorInstance;

    /// A type-cast [`Tensor`]
    type Cast: TensorInstance;

    /// A [`Tensor`] with an expanded dimension
    type Expand: TensorInstance;

    /// A [`Tensor`] flipped around one axis
    type Flip: TensorInstance;

    /// A reshaped [`Tensor`]
    type Reshape: TensorInstance;

    /// A [`Tensor`] slice
    type Slice: TensorInstance;

    /// A transposed [`Tensor`]
    type Transpose: TensorInstance;

    /// Broadcast this [`Tensor`] to the given `shape`.
    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast>;

    /// Cast this [`Tensor`] to the given `dtype`.
    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast>;

    /// Insert a new dimension of size 1 at the given `axis`.
    fn expand_dims(self, axis: usize) -> TCResult<Self::Expand>;

    /// Flip this [`Tensor`] around the given `axis`.
    fn flip(self, axis: usize) -> TCResult<Self::Flip>;

    /// Change the shape of this [`Tensor`].
    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape>;

    /// Return a slice of this [`Tensor`] with the given `bounds`.
    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    /// Transpose this [`Tensor`] by reordering its axes according to the given `permutation`.
    /// If no permutation is given, the axes will be reversed.
    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;
}

/// Unary [`Tensor`] operations
#[async_trait]
pub trait TensorUnary<D: Dir> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<D>;

    /// The return type of a unary operation
    type Unary: TensorInstance;

    /// Element-wise absolute value
    fn abs(&self) -> TCResult<Self::Unary>;

    /// Element-wise exponentiation
    fn exp(&self) -> TCResult<Self::Unary>;

    /// Element-wise natural logarithm
    fn ln(&self) -> TCResult<Self::Unary>;

    /// Element-wise round to the nearest integer
    fn round(&self) -> TCResult<Self::Unary>;

    /// Return `true` if all elements in this [`Tensor`] are nonzero.
    async fn all(self, txn: Self::Txn) -> TCResult<bool>;

    /// Return `true` if any element in this [`Tensor`] is nonzero.
    async fn any(self, txn: Self::Txn) -> TCResult<bool>;

    /// Element-wise logical not
    fn not(&self) -> TCResult<Self::Unary>;
}

/// Trigonometric [`Tensor`] operations
#[async_trait]
pub trait TensorTrig {
    /// The return type of a unary operation
    type Unary: TensorInstance;

    /// Element-wise arcsine
    fn asin(&self) -> TCResult<Self::Unary>;

    /// Element-wise sine
    fn sin(&self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic arcsine
    fn asinh(&self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic sine
    fn sinh(&self) -> TCResult<Self::Unary>;

    /// Element-wise arccosine
    fn acos(&self) -> TCResult<Self::Unary>;

    /// Element-wise cosine
    fn cos(&self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic arccosine
    fn acosh(&self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic cosine
    fn cosh(&self) -> TCResult<Self::Unary>;

    /// Element-wise arctangent
    fn atan(&self) -> TCResult<Self::Unary>;

    /// Element-wise tangent
    fn tan(&self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic tangent
    fn tanh(&self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic arctangent
    fn atanh(&self) -> TCResult<Self::Unary>;
}

fn trig_dtype(dtype: NumberType) -> NumberType {
    match dtype {
        NumberType::Int(it) => match it {
            IntType::I64 => FloatType::F64.into(),
            _ => FloatType::F32.into(),
        },
        NumberType::UInt(ut) => match ut {
            UIntType::U64 => FloatType::F64.into(),
            _ => FloatType::F32.into(),
        },
        other => other,
    }
}

/// The [`Class`] of [`Tensor`]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum TensorType {
    Dense,
    Sparse,
}

impl Class for TensorType {}

impl NativeClass for TensorType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 4 && &path[..3] == &PREFIX[..] {
            match path[3].as_str() {
                "dense" => Some(Self::Dense),
                "sparse" => Some(Self::Sparse),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        TCPathBuf::from(PREFIX).append(label(match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
        }))
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("type Tensor")
    }
}

/// An n-dimensional array of numbers which supports basic math and logic operations
#[derive(Clone)]
pub enum Tensor<FD, FS, D, T> {
    Dense(DenseTensor<FD, FS, D, T, DenseAccessor<FD, FS, D, T>>),
    Sparse(SparseTensor<FD, FS, D, T, SparseAccessor<FD, FS, D, T>>),
}

impl<FD, FS, D, T> Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    /// Get the [`Schema`] of this [`Tensor`]
    pub fn schema(&self) -> Schema {
        Schema {
            dtype: self.dtype(),
            shape: self.shape().clone(),
        }
    }
}

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> Instance for Tensor<FD, FS, D, T> {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Dense(_) => TensorType::Dense,
            Self::Sparse(_) => TensorType::Sparse,
        }
    }
}

impl<FD, FS, D, T> TensorAccess for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ndim(),
            Self::Sparse(sparse) => sparse.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Dense(dense) => dense.size(),
            Self::Sparse(sparse) => sparse.size(),
        }
    }
}

impl<FD, FS, D, T> TensorInstance for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Dense = Self;
    type Sparse = Self;

    fn into_dense(self) -> Self {
        match self {
            Self::Dense(dense) => Self::Dense(dense),
            Self::Sparse(sparse) => Self::Dense(sparse.into_dense().into_inner().accessor().into()),
        }
    }

    fn into_sparse(self) -> Self {
        match self {
            Self::Dense(dense) => Self::Sparse(dense.into_sparse().into_inner().accessor().into()),
            Self::Sparse(sparse) => Self::Sparse(sparse),
        }
    }
}

impl<FD, FS, D, T> TensorBoolean<Self> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Combine = Self;
    type LeftCombine = Self;

    fn and(self, other: Self) -> TCResult<Self::LeftCombine> {
        match self {
            Self::Dense(dense) => dense.and(other),
            Self::Sparse(sparse) => sparse.and(other),
        }
    }

    fn or(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.or(other),
            Self::Sparse(sparse) => sparse.or(other),
        }
    }

    fn xor(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.xor(other),
            Self::Sparse(sparse) => sparse.xor(other),
        }
    }
}

impl<FD, FS, D, T> TensorBooleanConst for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Combine = Self;
    type DenseCombine = Self;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.and_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.and_const(other).map(Self::from),
        }
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.or_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.or_const(other).map(Self::from),
        }
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.xor_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.xor_const(other).map(Self::from),
        }
    }
}

impl<FD, FS, D, T> TensorCompare<Self> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Compare = Self;
    type Dense = Self;

    fn eq(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.eq(other),
            Self::Sparse(sparse) => sparse.eq(other),
        }
    }

    fn gt(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.gt(other),
            Self::Sparse(sparse) => sparse.gt(other),
        }
    }

    fn gte(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.gte(other),
            Self::Sparse(sparse) => sparse.gte(other),
        }
    }

    fn lt(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.lt(other),
            Self::Sparse(sparse) => sparse.lt(other),
        }
    }

    fn lte(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.lte(other),
            Self::Sparse(sparse) => sparse.lte(other),
        }
    }

    fn ne(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.ne(other),
            Self::Sparse(sparse) => sparse.ne(other),
        }
    }
}

impl<FD, FS, D, T> TensorCompareConst for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Compare = Self;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(dense) => dense.eq_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.eq_const(other).map(Self::from),
        }
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(dense) => dense.gt_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.gt_const(other).map(Self::from),
        }
    }

    fn gte_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(dense) => dense.gte_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.gte_const(other).map(Self::from),
        }
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(dense) => dense.lt_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.lt_const(other).map(Self::from),
        }
    }

    fn lte_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(dense) => dense.lte_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.lte_const(other).map(Self::from),
        }
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(dense) => dense.ne_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.ne_const(other).map(Self::from),
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> TensorDiagonal<D> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
    SparseTable<FD, FS, D, T>: ReadValueAt<D, Txn = T>,
{
    type Txn = T;
    type Diagonal = Self;

    async fn diagonal(self, txn: Self::Txn) -> TCResult<Self::Diagonal> {
        match self {
            Self::Dense(dense) => dense.diagonal(txn).map_ok(Self::from).await,
            Self::Sparse(sparse) => sparse.diagonal(txn).map_ok(Self::from).await,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> TensorIO<D> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    async fn read_value(self, txn: Self::Txn, coord: Coord) -> TCResult<Number> {
        match self {
            Self::Dense(dense) => dense.read_value(txn, coord).await,
            Self::Sparse(sparse) => sparse.read_value(txn, coord).await,
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        debug!("Tensor::write_value {} {}", bounds, value);

        match self {
            Self::Dense(dense) => dense.write_value(txn_id, bounds, value).await,
            Self::Sparse(sparse) => sparse.write_value(txn_id, bounds, value).await,
        }
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        match self {
            Self::Dense(dense) => dense.write_value_at(txn_id, coord, value).await,
            Self::Sparse(sparse) => sparse.write_value_at(txn_id, coord, value).await,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> TensorDualIO<D, Self> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Txn = T;

    async fn write(self, txn: T, bounds: Bounds, value: Tensor<FD, FS, D, T>) -> TCResult<()> {
        debug!("Tensor::write {} to {}", value, bounds);

        match self {
            Self::Dense(this) => this.write(txn, bounds, value).await,
            Self::Sparse(this) => this.write(txn, bounds, value).await,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> TensorIndex<D> for Tensor<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Txn = T;
    type Index = Self;

    async fn argmax(self, txn: Self::Txn, axis: usize) -> TCResult<Self::Index> {
        match self {
            Self::Dense(dense) => dense.argmax(txn, axis).map_ok(Self::from).await,
            Self::Sparse(sparse) => sparse.argmax(txn, axis).map_ok(Self::from).await,
        }
    }

    async fn argmax_all(self, txn: Self::Txn) -> TCResult<u64> {
        match self {
            Self::Dense(dense) => dense.argmax_all(txn).await,
            Self::Sparse(sparse) => sparse.argmax_all(txn).await,
        }
    }
}

impl<FD, FS, D, T> TensorMath<D, Self> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Combine = Self;
    type LeftCombine = Self;

    fn add(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.add(other),
            Self::Sparse(this) => this.add(other),
        }
    }

    fn div(self, other: Self) -> TCResult<Self::LeftCombine> {
        match self {
            Self::Dense(this) => this.div(other),
            Self::Sparse(this) => this.div(other),
        }
    }

    fn log(self, base: Self) -> TCResult<Self::LeftCombine> {
        match self {
            Self::Dense(this) => this.log(base),
            Self::Sparse(this) => this.log(base),
        }
    }

    fn mul(self, other: Self) -> TCResult<Self::LeftCombine> {
        match self {
            Self::Dense(this) => this.mul(other),
            Self::Sparse(this) => this.mul(other),
        }
    }

    fn pow(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.pow(other),
            Self::Sparse(this) => this.pow(other),
        }
    }

    fn sub(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.sub(other),
            Self::Sparse(this) => this.sub(other),
        }
    }
}

impl<FD, FS, D, T> TensorMathConst for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Combine = Self;
    type DenseCombine = Self;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.add_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.add_const(other).map(Self::from),
        }
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.div_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.div_const(other).map(Self::from),
        }
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.log_const(base).map(Self::from),
            Self::Sparse(sparse) => sparse.log_const(base).map(Self::from),
        }
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.mul_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.mul_const(other).map(Self::from),
        }
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.pow_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.pow_const(other).map(Self::from),
        }
    }

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.sub_const(other).map(Self::from),
            Self::Sparse(sparse) => sparse.sub_const(other).map(Self::from),
        }
    }
}

impl<FD, FS, D, T> TensorReduce<D> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;
    type Reduce = Self;

    fn product(self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.product(axis).map(Self::from),
            Self::Sparse(sparse) => sparse.product(axis).map(Self::from),
        }
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.product_all(txn),
            Self::Sparse(sparse) => sparse.product_all(txn),
        }
    }

    fn sum(self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.sum(axis).map(Self::from),
            Self::Sparse(sparse) => sparse.sum(axis).map(Self::from),
        }
    }

    fn sum_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.sum_all(txn),
            Self::Sparse(sparse) => sparse.sum_all(txn),
        }
    }
}

impl<FD, FS, D, T> TensorTransform for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Broadcast = Self;
    type Cast = Self;
    type Expand = Self;
    type Flip = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> TCResult<Self> {
        if &shape == self.shape() {
            return Ok(self);
        }

        match self {
            Self::Dense(dense) => dense.broadcast(shape).map(Self::from),
            Self::Sparse(sparse) => sparse.broadcast(shape).map(Self::from),
        }
    }

    fn cast_into(self, dtype: NumberType) -> TCResult<Self> {
        if dtype == self.dtype() {
            return Ok(self);
        }

        match self {
            Self::Dense(dense) => TensorTransform::cast_into(dense, dtype).map(Self::from),
            Self::Sparse(sparse) => TensorTransform::cast_into(sparse, dtype).map(Self::from),
        }
    }

    fn expand_dims(self, axis: usize) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.expand_dims(axis).map(Self::from),
            Self::Sparse(sparse) => sparse.expand_dims(axis).map(Self::from),
        }
    }

    fn flip(self, axis: usize) -> TCResult<Self::Flip> {
        match self {
            Self::Dense(dense) => dense.flip(axis).map(Self::from),
            Self::Sparse(sparse) => sparse.flip(axis).map(Self::from),
        }
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        match self {
            Self::Dense(dense) => dense.reshape(shape).map(Self::from),
            Self::Sparse(sparse) => sparse.reshape(shape).map(Self::from),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        if bounds == Bounds::all(self.shape()) {
            return Ok(self);
        }

        match self {
            Self::Dense(dense) => dense.slice(bounds).map(Self::from),
            Self::Sparse(sparse) => sparse.slice(bounds).map(Self::from),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        if let Some(permutation) = &permutation {
            if permutation.iter().zip(0..self.ndim()).all(|(x, i)| x == &i) {
                debug!(
                    "no need to transpose {} with permutation {:?}",
                    self, permutation
                );
                return Ok(self);
            }
        }

        debug!("transpose {} with permutation {:?}", self, permutation);

        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(Self::from),
            Self::Sparse(sparse) => sparse.transpose(permutation).map(Self::from),
        }
    }
}

macro_rules! trig {
    ($fun:ident) => {
        fn $fun(&self) -> TCResult<Self> {
            match self {
                Self::Dense(dense) => dense.$fun().map(Self::from),
                Self::Sparse(sparse) => sparse.$fun().map(Self::from),
            }
        }
    };
}

impl<FD, FS, D, T> TensorTrig for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Unary = Self;

    trig! {asin}
    trig! {sin}
    trig! {asinh}
    trig! {sinh}

    trig! {acos}
    trig! {cos}
    trig! {acosh}
    trig! {cosh}

    trig! {atan}
    trig! {tan}
    trig! {atanh}
    trig! {tanh}
}

#[async_trait]
impl<FD, FS, D, T> TensorUnary<D> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;
    type Unary = Self;

    fn abs(&self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.abs().map(Self::from),
            Self::Sparse(sparse) => sparse.abs().map(Self::from),
        }
    }

    fn exp(&self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.exp().map(Self::from),
            Self::Sparse(sparse) => sparse.exp().map(Self::from),
        }
    }

    fn ln(&self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.ln().map(Self::from),
            Self::Sparse(sparse) => sparse.ln().map(Self::from),
        }
    }

    fn round(&self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.round().map(Self::from),
            Self::Sparse(sparse) => sparse.round().map(Self::from),
        }
    }

    async fn all(self, txn: T) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.all(txn).await,
            Self::Sparse(sparse) => sparse.all(txn).await,
        }
    }

    async fn any(self, txn: T) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.any(txn).await,
            Self::Sparse(sparse) => sparse.any(txn).await,
        }
    }

    fn not(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.not().map(Self::from),
            Self::Sparse(sparse) => sparse.not().map(Self::from),
        }
    }
}

impl<FD, FS, D, T, B> From<DenseTensor<FD, FS, D, T, B>> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    B: DenseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    fn from(dense: DenseTensor<FD, FS, D, T, B>) -> Self {
        Self::Dense(dense.into_inner().accessor().into())
    }
}

impl<FD, FS, D, T, A> From<SparseTensor<FD, FS, D, T, A>> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    A: SparseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    fn from(sparse: SparseTensor<FD, FS, D, T, A>) -> Self {
        Self::Sparse(sparse.into_inner().accessor().into())
    }
}

#[async_trait]
impl<FD, FS, D, T> de::FromStream for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    T: Transaction<D>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Context = T;

    async fn from_stream<De: de::Decoder>(txn: T, decoder: &mut De) -> Result<Self, De::Error> {
        decoder.decode_map(TensorVisitor::new(txn)).await
    }
}

struct TensorVisitor<FD, FS, D, T> {
    txn: T,
    dir: PhantomData<D>,
    dense: PhantomData<FD>,
    sparse: PhantomData<FS>,
}

impl<FD, FS, D, T> TensorVisitor<FD, FS, D, T> {
    fn new(txn: T) -> Self {
        Self {
            txn,
            dir: PhantomData,
            dense: PhantomData,
            sparse: PhantomData,
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> de::Visitor for TensorVisitor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    T: Transaction<D>,
    D::FileClass: From<BTreeType> + From<TensorType>,
{
    type Value = Tensor<FD, FS, D, T>;

    fn expecting() -> &'static str {
        "a Tensor"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let classpath = map
            .next_key::<TCPathBuf>(())
            .await?
            .ok_or_else(|| de::Error::custom("missing Tensor class"))?;

        let class = TensorType::from_path(&classpath)
            .ok_or_else(|| de::Error::invalid_type(classpath, "a Tensor class"))?;

        match class {
            TensorType::Dense => {
                map.next_value::<DenseTensor<FD, FS, D, T, BlockListFile<FD, FS, D, T>>>(self.txn)
                    .map_ok(Tensor::from)
                    .await
            }
            TensorType::Sparse => {
                map.next_value::<SparseTensor<FD, FS, D, T, SparseTable<FD, FS, D, T>>>(self.txn)
                    .map_ok(Tensor::from)
                    .await
            }
        }
    }
}

#[async_trait]
impl<'en, FD, FS, D, T> IntoView<'en, D> for Tensor<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;
    type View = TensorView<'en>;

    async fn into_view(self, txn: T) -> TCResult<Self::View> {
        match self {
            Tensor::Dense(dense) => dense.into_view(txn).map_ok(TensorView::Dense).await,
            Tensor::Sparse(sparse) => sparse.into_view(txn).map_ok(TensorView::Sparse).await,
        }
    }
}

/// A view of a [`Tensor`] at a given [`TxnId`], used in serialization
pub enum TensorView<'en> {
    Dense(dense::DenseTensorView<'en>),
    Sparse(sparse::SparseTensorView<'en>),
}

impl<'en> en::IntoStream<'en> for TensorView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Dense(view) => view.into_stream(encoder),
            Self::Sparse(view) => view.into_stream(encoder),
        }
    }
}

impl<FD, FS, D, T> fmt::Debug for Tensor<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl<FD, FS, D, T> fmt::Display for Tensor<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::File: AsType<FD> + AsType<FS>,
    D::FileClass: From<TensorType>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dense(dense) => fmt::Display::fmt(dense, f),
            Self::Sparse(sparse) => fmt::Display::fmt(sparse, f),
        }
    }
}

/// Broadcast the given `left` and `right` tensors into the same shape.
///
/// For rules of broadcasting, see:
/// [https://pytorch.org/docs/stable/notes/broadcasting.html](https://pytorch.org/docs/stable/notes/broadcasting.html)
pub fn broadcast<L, R>(left: L, right: R) -> TCResult<(L::Broadcast, R::Broadcast)>
where
    L: TensorAccess + TensorTransform,
    R: TensorAccess + TensorTransform,
{
    debug!(
        "broadcast tensors with shapes {}, {}",
        left.shape(),
        right.shape()
    );

    let mut left_shape = left.shape().to_vec();
    let mut right_shape = right.shape().to_vec();

    match (left_shape.len(), right_shape.len()) {
        (l, r) if l < r => {
            for _ in 0..(r - l) {
                left_shape.insert(0, 1);
            }
        }
        (l, r) if r < l => {
            for _ in 0..(l - r) {
                right_shape.insert(0, 1);
            }
        }
        _ => {}
    }

    let mut shape = Vec::with_capacity(left_shape.len());
    for (l, r) in left_shape.iter().zip(right_shape.iter()) {
        if l == r || *l == 1 {
            shape.push(*r);
        } else if *r == 1 {
            shape.push(*l)
        } else {
            return Err(TCError::unsupported(format!(
                "cannot broadcast dimension {} into {}",
                l, r
            )));
        }
    }

    let shape = Shape::from(shape);
    debug!("broadcast shape is {}", shape);
    Ok((left.broadcast(shape.clone())?, right.broadcast(shape)?))
}

#[derive(Clone)]
struct Phantom<FD, FS, D, T> {
    dense: PhantomData<FD>,
    sparse: PhantomData<FS>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<FD, FS, D, T> Default for Phantom<FD, FS, D, T> {
    #[inline]
    fn default() -> Self {
        Self {
            dense: PhantomData,
            sparse: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

async fn tile<D, T, I, O>(txn: T, input: I, output: O, multiples: Vec<u64>) -> TCResult<O>
where
    D: Dir,
    T: Transaction<D>,
    I: TensorAccess + Clone,
    O: TensorAccess + TensorDualIO<D, I, Txn = T> + Clone,
{
    let max_usize = usize::MAX as u64;
    let input_shape = input.shape();

    if input_shape.iter().any(|dim| dim >= &max_usize) {
        return Err(TCError::unsupported(format!(
            "Tensor of shape {} is too large to tile",
            input_shape
        )));
    }

    let slices = input_shape
        .iter()
        .zip(&multiples)
        .map(|(dim, m)| (0..(dim * m)).step_by(*dim as usize));

    let mut writes = Vec::with_capacity((output.size() / input.size()) as usize);
    for slice in Itertools::multi_cartesian_product(slices) {
        let bounds = slice
            .into_iter()
            .zip(input_shape.iter())
            .map(|(i, dim)| AxisBounds::In(i..(i + *dim)))
            .collect();

        writes.push(output.clone().write(txn.clone(), bounds, input.clone()));
    }

    futures::stream::iter(writes)
        .buffer_unordered(num_cpus::get())
        .try_fold((), |(), ()| futures::future::ready(Ok(())))
        .await?;

    Ok(output)
}

#[inline]
fn coord_bounds(shape: &[u64]) -> Vec<u64> {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}

#[inline]
fn needs_transpose(permutation: &Option<Vec<usize>>) -> bool {
    if let Some(axes) = permutation {
        !axes.iter().enumerate().all(|(i, x)| x == &i)
    } else {
        true
    }
}
