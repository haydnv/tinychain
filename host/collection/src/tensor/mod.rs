/// A [`Tensor`], an n-dimensional array of [`Number`]s which supports basic math and logic
use std::fmt;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};

use tc_error::*;
use tc_transact::{Transaction, TxnId};
use tc_value::{Number, NumberType, ValueType};
use tcgeneric::{
    label, path_label, Class, NativeClass, PathLabel, PathSegment, TCBoxTryFuture, TCPathBuf,
};

pub use fensor::{AxisRange, Range, Shape};

pub mod dense;
pub mod sparse;

/// A [`Tensor`] coordinate
pub type Coord = Vec<u64>;

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    dtype: NumberType,
    shape: Shape,
}

impl<'a, D: Digest> Hash<D> for &'a Schema {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash((&self.shape, ValueType::from(self.dtype).path()))
    }
}

impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "tensor of type {:?} with shape {:?}",
            self.dtype, self.shape
        )
    }
}

#[async_trait]
impl de::FromStream for Schema {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let (classpath, shape): (TCPathBuf, Shape) =
            de::FromStream::from_stream(cxt, decoder).await?;

        if let Some(ValueType::Number(dtype)) = ValueType::from_path(&classpath) {
            Ok(Self { shape, dtype })
        } else {
            Err(de::Error::invalid_value("a Number type", classpath))
        }
    }
}

impl<'en> en::IntoStream<'en> for Schema {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((ValueType::from(self.dtype).path(), self.shape), encoder)
    }
}

impl<'en> en::ToStream<'en> for Schema {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((ValueType::from(self.dtype).path(), &self.shape), encoder)
    }
}

const PREFIX: PathLabel = path_label(&["state", "collection", "tensor"]);

/// The [`Class`] of a [`Tensor`]
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

impl fmt::Debug for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("type Tensor")
    }
}

/// A [`Tensor`] instance
pub trait TensorInstance: fensor::TensorInstance {
    // /// A dense representation of this [`Tensor`]
    // type Dense: TensorInstance;

    // /// A sparse representation of this [`Tensor`]
    // type Sparse: TensorInstance;

    // /// Return a dense representation of this [`Tensor`].
    // fn into_dense(self) -> Self::Dense;

    // /// Return a sparse representation of this [`Tensor`].
    // fn into_sparse(self) -> Self::Sparse;
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
pub trait TensorDiagonal<FE> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<FE>;

    /// The type of [`Tensor`] returned by `diagonal`
    type Diagonal: TensorInstance;

    async fn diagonal(self, txn: Self::Txn) -> TCResult<Self::Diagonal>;
}

/// [`Tensor`] I/O operations
#[async_trait]
pub trait TensorIO<FE> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<FE>;

    /// Read a single value from this [`Tensor`].
    async fn read_value(self, txn: Self::Txn, coord: Coord) -> TCResult<Number>;

    /// Write a single value to the slice of this [`Tensor`] with the given [`Range`].
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()>;

    /// Overwrite a single element of this [`Tensor`].
    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

/// [`Tensor`] I/O operations which accept another [`Tensor`] as an argument
#[async_trait]
pub trait TensorDualIO<FE, O> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<FE>;

    /// Overwrite the slice of this [`Tensor`] given by [`Range`] with the given `value`.
    async fn write(self, txn: Self::Txn, range: Range, value: O) -> TCResult<()>;
}

/// [`Tensor`] indexing operations
#[async_trait]
pub trait TensorIndex<FE> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<FE>;

    /// The type of [`Tensor`] returned by `argmax`.
    type Index: TensorInstance;

    /// Return the indices of the maximum values in this [`Tensor`] along the given `axis`.
    async fn argmax(self, txn: Self::Txn, axis: usize) -> TCResult<Self::Index>;

    /// Return the offset of the maximum value in this [`Tensor`].
    async fn argmax_all(self, txn: Self::Txn) -> TCResult<u64>;
}

/// [`Tensor`] math operations
pub trait TensorMath<O> {
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

    fn is_persistent(&self) -> bool;
}

/// [`Tensor`] reduction operations
pub trait TensorReduce<FE> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<FE>;

    /// The result type of a reduce operation
    type Reduce: TensorInstance;

    /// Return the maximum of this [`Tensor`] along the given `axis`.
    fn max(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce>;

    /// Return the maximum element in this [`Tensor`].
    fn max_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number>;

    /// Return the minimum of this [`Tensor`] along the given `axis`.
    fn min(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce>;

    /// Return the minimum element in this [`Tensor`].
    fn min_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number>;

    /// Return the product of this [`Tensor`] along the given `axis`.
    fn product(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce>;

    /// Return the product of all elements in this [`Tensor`].
    fn product_all(&self, txn: Self::Txn) -> TCBoxTryFuture<Number>;

    /// Return the sum of this [`Tensor`] along the given `axis`.
    fn sum(self, axis: usize, keepdims: bool) -> TCResult<Self::Reduce>;

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

    /// Return a slice of this [`Tensor`] with the given `range`.
    fn slice(self, range: Range) -> TCResult<Self::Slice>;

    /// Transpose this [`Tensor`] by reordering its axes according to the given `permutation`.
    /// If no permutation is given, the axes will be reversed.
    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;
}

/// Unary [`Tensor`] operations
#[async_trait]
pub trait TensorUnary<FE> {
    /// The type of [`Transaction`] to expect
    type Txn: Transaction<FE>;

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
