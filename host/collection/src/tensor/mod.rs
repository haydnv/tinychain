/// A [`Tensor`], an n-dimensional array of [`Number`]s which supports basic math and logic
use std::fmt;
use std::ops::{Div, Rem};

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use collate::Collator;
use destream::{de, en};

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::TxnId;
use tc_value::{Number, NumberType, ValueType};
use tcgeneric::{
    label, path_label, Class, NativeClass, PathLabel, PathSegment, TCPathBuf, ThreadSafe,
};

pub use shape::{AxisRange, Range, Shape};

mod block;
pub mod dense;
pub mod shape;
pub mod sparse;
mod transform;

const PREFIX: PathLabel = path_label(&["state", "collection", "tensor"]);

#[cfg(debug_assertions)]
const IDEAL_BLOCK_SIZE: usize = 24;

#[cfg(not(debug_assertions))]
const IDEAL_BLOCK_SIZE: usize = 65_536;

pub type Axes = Vec<usize>;

/// A [`Tensor`] coordinate
pub type Coord = Vec<u64>;

pub type Strides = Vec<u64>;

type Semaphore = tc_transact::lock::Semaphore<Collator<u64>, Range>;

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

#[async_trait]
pub trait TensorPermitRead: Send + Sync {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>>;
}

#[async_trait]
pub trait TensorPermitWrite: Send + Sync {
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>>;
}

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
pub trait TensorInstance: ThreadSafe {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn shape(&self) -> &Shape;

    fn size(&self) -> u64 {
        self.shape().iter().product()
    }
}

impl<T: TensorInstance> TensorInstance for Box<T> {
    fn dtype(&self) -> NumberType {
        (&**self).dtype()
    }

    fn shape(&self) -> &Shape {
        (&**self).shape()
    }
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

pub trait TensorCast {
    type Cast;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast>;
}

/// Tensor comparison operations
pub trait TensorCompare<O> {
    /// The result of a comparison operation
    type Compare: TensorInstance;

    /// Element-wise equality
    fn eq(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise greater-than
    fn gt(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise greater-or-equal
    fn ge(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise less-than
    fn lt(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise less-or-equal
    fn le(self, other: O) -> TCResult<Self::Compare>;

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
    fn ge_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise less-than
    fn lt_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise less-or-equal
    fn le_const(self, other: Number) -> TCResult<Self::Compare>;

    /// Element-wise not-equal
    fn ne_const(self, other: Number) -> TCResult<Self::Compare>;
}

/// Methods to convert between a sparse an dense [`Tensor`]
pub trait TensorConvert: ThreadSafe {
    /// A dense representation of this [`Tensor`]
    type Dense: TensorInstance;

    // /// A sparse representation of this [`Tensor`]
    // type Sparse: TensorInstance;

    /// Return a dense representation of this [`Tensor`].
    fn into_dense(self) -> Self::Dense;

    // /// Return a sparse representation of this [`Tensor`].
    // fn into_sparse(self) -> Self::Sparse;
}

/// [`Tensor`] linear algebra operations
pub trait TensorDiagonal {
    /// The type of [`Tensor`] returned by `diagonal`
    type Diagonal: TensorInstance;

    fn diagonal(self) -> TCResult<Self::Diagonal>;
}

/// [`Tensor`] I/O operations
#[async_trait]
pub trait TensorIO {
    /// Read a single value from this [`Tensor`].
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number>;

    /// Write a single value to the slice of this [`Tensor`] with the given [`Range`].
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()>;

    /// Overwrite a single element of this [`Tensor`].
    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

/// [`Tensor`] I/O operations which accept another [`Tensor`] as an argument
#[async_trait]
pub trait TensorDualIO<O> {
    /// Overwrite the slice of this [`Tensor`] given by [`Range`] with the given `value`.
    async fn write(self, txn_id: TxnId, range: Range, value: O) -> TCResult<()>;
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

/// [`Tensor`] reduction operations
#[async_trait]
pub trait TensorReduce {
    /// The result type of a reduce operation
    type Reduce: TensorInstance;

    /// Return `true` if all elements in this [`Tensor`] are nonzero.
    async fn all(self, txn_id: TxnId) -> TCResult<bool>;

    /// Return `true` if any element in this [`Tensor`] is nonzero.
    async fn any(self, txn_id: TxnId) -> TCResult<bool>;

    /// Return the maximum of this [`Tensor`] along the given `axis`.
    fn max(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce>;

    /// Return the maximum element in this [`Tensor`].
    async fn max_all(self, txn_id: TxnId) -> TCResult<Number>;

    /// Return the minimum of this [`Tensor`] along the given `axis`.
    fn min(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce>;

    /// Return the minimum element in this [`Tensor`].
    async fn min_all(self, txn_id: TxnId) -> TCResult<Number>;

    /// Return the product of this [`Tensor`] along the given `axis`.
    fn product(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce>;

    /// Return the product of all elements in this [`Tensor`].
    async fn product_all(self, txn_id: TxnId) -> TCResult<Number>;

    /// Return the sum of this [`Tensor`] along the given `axis`.
    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce>;

    /// Return the sum of all elements in this [`Tensor`].
    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number>;
}

/// [`Tensor`] transforms
pub trait TensorTransform {
    /// A broadcast [`Tensor`]
    type Broadcast: TensorInstance;

    /// A [`Tensor`] with an expanded dimension
    type Expand: TensorInstance;

    /// A reshaped [`Tensor`]
    type Reshape: TensorInstance;

    /// A [`Tensor`] slice
    type Slice: TensorInstance;

    /// A transposed [`Tensor`]
    type Transpose: TensorInstance;

    /// Broadcast this [`Tensor`] to the given `shape`.
    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast>;

    /// Insert a new dimension of size 1 at each of the given `axes`.
    fn expand(self, axes: Axes) -> TCResult<Self::Expand>;

    /// Change the shape of this [`Tensor`].
    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape>;

    /// Return a slice of this [`Tensor`] with the given `range`.
    fn slice(self, range: Range) -> TCResult<Self::Slice>;

    /// Transpose this [`Tensor`] by reordering its axes according to the given `permutation`.
    /// If no permutation is given, the axes will be reversed.
    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;
}

/// Trigonometric [`Tensor`] operations
#[async_trait]
pub trait TensorTrig {
    /// The return type of a unary operation
    type Unary: TensorInstance;

    /// Element-wise arcsine
    fn asin(self) -> TCResult<Self::Unary>;

    /// Element-wise sine
    fn sin(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic arcsine
    fn asinh(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic sine
    fn sinh(self) -> TCResult<Self::Unary>;

    /// Element-wise arccosine
    fn acos(self) -> TCResult<Self::Unary>;

    /// Element-wise cosine
    fn cos(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic arccosine
    fn acosh(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic cosine
    fn cosh(self) -> TCResult<Self::Unary>;

    /// Element-wise arctangent
    fn atan(self) -> TCResult<Self::Unary>;

    /// Element-wise tangent
    fn tan(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic tangent
    fn tanh(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic arctangent
    fn atanh(self) -> TCResult<Self::Unary>;
}

/// Unary [`Tensor`] operations
pub trait TensorUnary {
    /// The return type of a unary operation
    type Unary: TensorInstance;

    /// Element-wise absolute value
    fn abs(self) -> TCResult<Self::Unary>;

    /// Element-wise exponentiation
    fn exp(self) -> TCResult<Self::Unary>;

    /// Element-wise natural logarithm
    fn ln(self) -> TCResult<Self::Unary>;

    /// Element-wise round to the nearest integer
    fn round(self) -> TCResult<Self::Unary>;
}

/// Unary [`Tensor`] operations
pub trait TensorUnaryBoolean {
    /// The return type of a unary operation
    type Unary: TensorInstance;

    /// Element-wise logical not
    fn not(self) -> TCResult<Self::Unary>;
}

#[inline]
fn coord_of<T: Copy + Div<Output = T> + Rem<Output = T>>(
    offset: T,
    strides: &[T],
    shape: &[T],
) -> Vec<T> {
    strides
        .iter()
        .copied()
        .map(|stride| offset / stride)
        .zip(shape.iter().copied())
        .map(|(axis_offset, dim)| axis_offset % dim)
        .collect()
}

#[inline]
fn offset_of(coord: Coord, shape: &[u64]) -> u64 {
    let strides = shape.iter().enumerate().map(|(x, dim)| {
        if *dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    coord.into_iter().zip(strides).map(|(i, dim)| i * dim).sum()
}

#[inline]
fn strides_for(shape: &[u64], ndim: usize) -> Strides {
    debug_assert!(ndim >= shape.len());

    let zeros = std::iter::repeat(0).take(ndim - shape.len());

    let strides = shape.iter().enumerate().map(|(x, dim)| {
        if *dim == 1 {
            0
        } else {
            shape.iter().rev().take(shape.len() - 1 - x).product()
        }
    });

    zeros.chain(strides).collect()
}

#[inline]
fn validate_order(order: &[usize], ndim: usize) -> bool {
    if order.is_empty() {
        true
    } else {
        order.len() == ndim && order.iter().all(|x| x < &ndim)
    }
}
