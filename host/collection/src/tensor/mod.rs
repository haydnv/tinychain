/// A [`Tensor`], an n-dimensional array of [`Number`]s which supports basic math and logic
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Div, Rem};

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use collate::Collator;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::AsType;

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::{fs, IntoView, Transact, Transaction, TxnId};
use tc_value::{Number, NumberType, ValueType};
use tcgeneric::{
    label, path_label, Class, Instance, Label, NativeClass, PathLabel, PathSegment, TCPathBuf,
    ThreadSafe,
};

pub use dense::{DenseBase, DenseCacheFile, DenseView};
pub use shape::{AxisRange, Range, Shape};
pub use sparse::{Node, SparseBase, SparseView};

mod block;
mod complex;
pub mod dense;
pub mod shape;
pub mod sparse;
mod transform;
mod view;

const REAL: Label = label("re");
const IMAG: Label = label("im");

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

impl From<(NumberType, Shape)> for Schema {
    fn from(schema: (NumberType, Shape)) -> Self {
        let (dtype, shape) = schema;
        Self { dtype, shape }
    }
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
pub trait TensorInstance: ThreadSafe + Sized {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn shape(&self) -> &Shape;

    fn size(&self) -> u64 {
        self.shape().iter().product()
    }

    fn schema(&self) -> Schema {
        Schema::from((self.dtype(), self.shape().clone()))
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

    /// Logical and
    fn and_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Logical or
    fn or_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Logical xor
    fn xor_const(self, other: Number) -> TCResult<Self::Combine>;
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

    /// A sparse representation of this [`Tensor`]
    type Sparse: TensorInstance;

    /// Return a dense representation of this [`Tensor`].
    fn into_dense(self) -> Self::Dense;

    /// Return a sparse representation of this [`Tensor`].
    fn into_sparse(self) -> Self::Sparse;
}

/// [`Tensor`] linear algebra operations
pub trait TensorDiagonal {
    /// The type of [`Tensor`] returned by `diagonal`
    type Diagonal: TensorInstance;

    fn diagonal(self) -> TCResult<Self::Diagonal>;
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

    /// Add a constant to this tensor
    fn add_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Divide `self` by `other`.
    fn div_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Element-wise logarithm
    fn log_const(self, base: Number) -> TCResult<Self::Combine>;

    /// Multiply `self` by `other`.
    fn mul_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Raise `self` to the power `other`.
    fn pow_const(self, other: Number) -> TCResult<Self::Combine>;

    /// Subtract `other` from `self`.
    fn sub_const(self, other: Number) -> TCResult<Self::Combine>;
}

/// [`Tensor`] read operations
#[async_trait]
pub trait TensorRead {
    /// Read a single value from this [`Tensor`].
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number>;
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

    /// Element-wise hyperbolic sine
    fn sinh(self) -> TCResult<Self::Unary>;

    /// Element-wise arccosine
    fn acos(self) -> TCResult<Self::Unary>;

    /// Element-wise cosine
    fn cos(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic cosine
    fn cosh(self) -> TCResult<Self::Unary>;

    /// Element-wise arctangent
    fn atan(self) -> TCResult<Self::Unary>;

    /// Element-wise tangent
    fn tan(self) -> TCResult<Self::Unary>;

    /// Element-wise hyperbolic tangent
    fn tanh(self) -> TCResult<Self::Unary>;
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

/// [`Tensor`] write operations
#[async_trait]
pub trait TensorWrite {
    /// Write a single value to the slice of this [`Tensor`] with the given [`Range`].
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()>;

    /// Overwrite a single element of this [`Tensor`].
    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

/// [`Tensor`] I/O operations which accept another [`Tensor`] as an argument
#[async_trait]
pub trait TensorWriteDual<O> {
    /// Overwrite the slice of this [`Tensor`] given by [`Range`] with the given `value`.
    async fn write(self, txn_id: TxnId, range: Range, value: O) -> TCResult<()>;
}

/// A dense [`Tensor`]
pub enum Dense<Txn, FE> {
    Base(DenseBase<Txn, FE>),
    View(DenseView<Txn, FE>),
}

impl<Txn, FE> Dense<Txn, FE> {
    pub fn into_view(self) -> DenseView<Txn, FE> {
        self.into()
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> TensorInstance for Dense<Txn, FE> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Base(base) => base.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Base(base) => base.shape(),
            Self::View(view) => view.shape(),
        }
    }
}

#[async_trait]
impl<Txn, FE> TensorRead for Dense<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        match self {
            Self::Base(base) => base.read_value(txn_id, coord).await,
            Self::View(view) => view.read_value(txn_id, coord).await,
        }
    }
}

#[async_trait]
impl<Txn, FE> TensorWrite for Dense<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
        if let Self::Base(base) = self {
            base.write_value(txn_id, range, value).await
        } else {
            Err(bad_request!("cannot write to {:?}", self))
        }
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        if let Self::Base(base) = self {
            base.write_value_at(txn_id, coord, value).await
        } else {
            Err(bad_request!("cannot write to {:?}", self))
        }
    }
}

#[async_trait]
impl<Txn, FE> TensorWriteDual<Self> for Dense<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn write(self, txn_id: TxnId, range: Range, value: Self) -> TCResult<()> {
        if let Self::Base(base) = self {
            base.write(txn_id, range, value.into()).await
        } else {
            Err(bad_request!("cannot write to {:?}", self))
        }
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for Dense<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Txn = Txn;
    type View = view::DenseView;

    async fn into_view(self, txn: Self::Txn) -> TCResult<view::DenseView> {
        view::DenseView::read_from(self, *txn.id()).await
    }
}

impl<Txn, FE> From<DenseView<Txn, FE>> for Dense<Txn, FE> {
    fn from(view: DenseView<Txn, FE>) -> Self {
        Self::View(view)
    }
}

impl<Txn, FE> From<Dense<Txn, FE>> for DenseView<Txn, FE> {
    fn from(dense: Dense<Txn, FE>) -> Self {
        match dense {
            Dense::Base(base) => base.into(),
            Dense::View(view) => view,
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for Dense<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Base(base) => base.fmt(f),
            Self::View(view) => view.fmt(f),
        }
    }
}

/// A sparse [`Tensor`]
pub enum Sparse<Txn, FE> {
    Base(SparseBase<Txn, FE>),
    View(SparseView<Txn, FE>),
}

impl<Txn, FE> Sparse<Txn, FE> {
    pub fn into_view(self) -> SparseView<Txn, FE> {
        self.into()
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> TensorInstance for Sparse<Txn, FE> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Base(base) => base.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Base(base) => base.shape(),
            Self::View(view) => view.shape(),
        }
    }
}

#[async_trait]
impl<Txn: Transaction<FE>, FE: DenseCacheFile + AsType<Node>> TensorRead for Sparse<Txn, FE> {
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        match self {
            Self::Base(base) => base.read_value(txn_id, coord).await,
            Self::View(view) => view.read_value(txn_id, coord).await,
        }
    }
}

#[async_trait]
impl<Txn: Transaction<FE>, FE: DenseCacheFile + AsType<Node>> TensorWrite for Sparse<Txn, FE> {
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
        if let Self::Base(base) = self {
            base.write_value(txn_id, range, value).await
        } else {
            Err(bad_request!("cannot write to {:?}", self))
        }
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        if let Self::Base(base) = self {
            base.write_value_at(txn_id, coord, value).await
        } else {
            Err(bad_request!("cannot write to {:?}", self))
        }
    }
}

#[async_trait]
impl<Txn, FE> TensorWriteDual<Self> for Sparse<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn write(self, txn_id: TxnId, range: Range, value: Self) -> TCResult<()> {
        if let Self::Base(base) = self {
            base.write(txn_id, range, value.into()).await
        } else {
            Err(bad_request!("cannot write to {:?}", self))
        }
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for Sparse<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Txn = Txn;
    type View = view::SparseView;

    async fn into_view(self, txn: Self::Txn) -> TCResult<view::SparseView> {
        view::SparseView::read_from(self, *txn.id()).await
    }
}

impl<Txn, FE> From<SparseView<Txn, FE>> for Sparse<Txn, FE> {
    fn from(view: SparseView<Txn, FE>) -> Self {
        Self::View(view)
    }
}

impl<Txn, FE> From<Sparse<Txn, FE>> for SparseView<Txn, FE> {
    fn from(sparse: Sparse<Txn, FE>) -> Self {
        match sparse {
            Sparse::Base(base) => base.into(),
            Sparse::View(view) => view.into(),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for Sparse<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Base(base) => base.fmt(f),
            Self::View(view) => view.fmt(f),
        }
    }
}

/// An n-dimensional array of numbers which supports transactional reads and writes
pub enum Tensor<Txn, FE> {
    Dense(Dense<Txn, FE>),
    Sparse(Sparse<Txn, FE>),
}

impl<Txn, FE> Tensor<Txn, FE> {
    pub fn into_view(self) -> TensorView<Txn, FE> {
        self.into()
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> TensorInstance for Tensor<Txn, FE> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }
}

impl<Txn, FE> TensorBoolean<Self> for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;
    type LeftCombine = Self;

    fn and(self, other: Self) -> TCResult<Self::LeftCombine> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().and(that.into()).map(Self::from),

                Self::Sparse(that) => that
                    .into_view()
                    .and(this.into_view().into_sparse())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .and(that.into_view().into_sparse())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().and(that.into()).map(Self::from),
            },
        }
    }

    fn or(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().or(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .or(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .or(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().or(that.into()).map(Self::from),
            },
        }
    }

    fn xor(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().xor(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .xor(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .xor(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().xor(that.into()).map(Self::from),
            },
        }
    }
}

impl<Txn, FE> TensorBooleanConst for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;

    fn and_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().and_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().and_const(other).map(Self::from),
        }
    }

    fn or_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().or_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().or_const(other).map(Self::from),
        }
    }

    fn xor_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().xor_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().xor_const(other).map(Self::from),
        }
    }
}

impl<Txn, FE> TensorCast for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Cast = Self;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast> {
        match self {
            Self::Dense(this) => TensorCast::cast_into(this.into_view(), dtype).map(Self::from),
            Self::Sparse(this) => TensorCast::cast_into(this.into_view(), dtype).map(Self::from),
        }
    }
}

impl<Txn, FE> TensorCompare<Self> for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Compare = Self;

    fn eq(self, other: Self) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().eq(that.into()).map(Self::from),

                Self::Sparse(that) => that
                    .into_view()
                    .eq(this.into_view().into_sparse())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .eq(that.into_view().into_sparse())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().eq(that.into()).map(Self::from),
            },
        }
    }

    fn gt(self, other: Self) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().gt(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .gt(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .gt(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().gt(that.into()).map(Self::from),
            },
        }
    }

    fn ge(self, other: Self) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().ge(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .ge(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .ge(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().ge(that.into()).map(Self::from),
            },
        }
    }

    fn lt(self, other: Self) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().lt(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .lt(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .lt(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().lt(that.into()).map(Self::from),
            },
        }
    }

    fn le(self, other: Self) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().le(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .le(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .le(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().le(that.into()).map(Self::from),
            },
        }
    }

    fn ne(self, other: Self) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().ne(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .ne(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .ne(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().ne(that.into()).map(Self::from),
            },
        }
    }
}

impl<Txn, FE> TensorCompareConst for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Compare = Self;

    fn eq_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => this.into_view().eq_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().eq_const(other).map(Self::from),
        }
    }

    fn gt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => this.into_view().gt_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().gt_const(other).map(Self::from),
        }
    }

    fn ge_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => this.into_view().ge_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().ge_const(other).map(Self::from),
        }
    }

    fn lt_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => this.into_view().ge_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().eq_const(other).map(Self::from),
        }
    }

    fn le_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => this.into_view().ge_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().ge_const(other).map(Self::from),
        }
    }

    fn ne_const(self, other: Number) -> TCResult<Self::Compare> {
        match self {
            Self::Dense(this) => this.into_view().ne_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().ne_const(other).map(Self::from),
        }
    }
}

impl<Txn, FE> TensorConvert for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Dense = Dense<Txn, FE>;
    type Sparse = Sparse<Txn, FE>;

    fn into_dense(self) -> Dense<Txn, FE> {
        match self {
            Self::Dense(this) => this,
            Self::Sparse(this) => Dense::View(this.into_view().into_dense()),
        }
    }

    fn into_sparse(self) -> Sparse<Txn, FE> {
        match self {
            Self::Dense(this) => Sparse::View(this.into_view().into_sparse()),
            Self::Sparse(this) => this,
        }
    }
}

impl<Txn, FE> TensorDiagonal for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Diagonal = Self;

    fn diagonal(self) -> TCResult<Self::Diagonal> {
        match self {
            Self::Dense(dense) => dense.into_view().diagonal().map(Self::from),

            Self::Sparse(sparse) => Err(not_implemented!("diagonal of {:?}", sparse)),
        }
    }
}

impl<Txn, FE> TensorMath<Self> for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;
    type LeftCombine = Self;

    fn add(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().add(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .add(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .add(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().add(that.into_view()).map(Self::from),
            },
        }
    }

    fn div(self, other: Self) -> TCResult<Self::LeftCombine> {
        if let Self::Dense(that) = other {
            match self {
                Self::Dense(this) => this.into_view().div(that.into()).map(Self::from),

                Self::Sparse(this) => this
                    .into_view()
                    .div(that.into_view().into_sparse())
                    .map(Self::from),
            }
        } else {
            Err(bad_request!("cannot divide by {other:?}"))
        }
    }

    fn log(self, base: Self) -> TCResult<Self::LeftCombine> {
        if let Self::Dense(that) = base {
            match self {
                Self::Dense(this) => this.into_view().log(that.into()).map(Self::from),

                Self::Sparse(this) => this
                    .into_view()
                    .log(that.into_view().into_sparse())
                    .map(Self::from),
            }
        } else {
            Err(bad_request!("log base {base:?} is undefined"))
        }
    }

    fn mul(self, other: Self) -> TCResult<Self::LeftCombine> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().mul(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .into_sparse()
                    .mul(that.into())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .mul(that.into_view().into_sparse())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().mul(that.into()).map(Self::from),
            },
        }
    }

    fn pow(self, other: Self) -> TCResult<Self::LeftCombine> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().pow(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .pow(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .pow(that.into_view().into_sparse())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().pow(that.into()).map(Self::from),
            },
        }
    }

    fn sub(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => match other {
                Self::Dense(that) => this.into_view().sub(that.into()).map(Self::from),

                Self::Sparse(that) => this
                    .into_view()
                    .sub(that.into_view().into_dense())
                    .map(Self::from),
            },
            Self::Sparse(this) => match other {
                Self::Dense(that) => this
                    .into_view()
                    .into_dense()
                    .sub(that.into())
                    .map(Self::from),

                Self::Sparse(that) => this.into_view().sub(that.into()).map(Self::from),
            },
        }
    }
}

impl<Txn, FE> TensorMathConst for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Combine = Self;

    fn add_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().add_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().add_const(other).map(Self::from),
        }
    }

    fn div_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().div_const(other).map(Self::from),

            Self::Sparse(this) => this.into_view().div_const(other).map(Self::from),
        }
    }

    fn log_const(self, base: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().log_const(base).map(Self::from),
            Self::Sparse(this) => this.into_view().log_const(base).map(Self::from),
        }
    }

    fn mul_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().mul_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().mul_const(other).map(Self::from),
        }
    }

    fn pow_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().pow_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().pow_const(other).map(Self::from),
        }
    }

    fn sub_const(self, other: Number) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.into_view().sub_const(other).map(Self::from),
            Self::Sparse(this) => this.into_view().sub_const(other).map(Self::from),
        }
    }
}

#[async_trait]
impl<Txn, FE> TensorRead for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn read_value(self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        match self {
            Self::Dense(dense) => dense.read_value(txn_id, coord).await,
            Self::Sparse(sparse) => sparse.read_value(txn_id, coord).await,
        }
    }
}

#[async_trait]
impl<Txn, FE> TensorReduce for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Reduce = Self;

    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::Dense(this) => this.into_view().all(txn_id).await,
            Self::Sparse(this) => this.into_view().all(txn_id).await,
        }
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        match self {
            Self::Dense(this) => this.into_view().any(txn_id).await,
            Self::Sparse(this) => this.into_view().any(txn_id).await,
        }
    }

    fn max(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(this) => this.into_view().max(axes, keepdims).map(Self::from),
            Self::Sparse(this) => this.into_view().max(axes, keepdims).map(Self::from),
        }
    }

    async fn max_all(self, txn_id: TxnId) -> TCResult<Number> {
        match self {
            Self::Dense(this) => this.into_view().max_all(txn_id).await,
            Self::Sparse(this) => this.into_view().max_all(txn_id).await,
        }
    }

    fn min(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(this) => this.into_view().min(axes, keepdims).map(Self::from),
            Self::Sparse(this) => this.into_view().min(axes, keepdims).map(Self::from),
        }
    }

    async fn min_all(self, txn_id: TxnId) -> TCResult<Number> {
        match self {
            Self::Dense(this) => this.into_view().min_all(txn_id).await,
            Self::Sparse(this) => this.into_view().min_all(txn_id).await,
        }
    }

    fn product(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(this) => this.into_view().product(axes, keepdims).map(Self::from),
            Self::Sparse(this) => this.into_view().product(axes, keepdims).map(Self::from),
        }
    }

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number> {
        match self {
            Self::Dense(this) => this.into_view().product_all(txn_id).await,
            Self::Sparse(this) => this.into_view().product_all(txn_id).await,
        }
    }

    fn sum(self, axes: Axes, keepdims: bool) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(this) => this.into_view().sum(axes, keepdims).map(Self::from),

            Self::Sparse(this) => this.into_view().sum(axes, keepdims).map(Self::from),
        }
    }

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number> {
        match self {
            Self::Dense(this) => this.into_view().sum_all(txn_id).await,
            Self::Sparse(this) => this.into_view().sum_all(txn_id).await,
        }
    }
}

impl<Txn, FE> TensorTransform for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Broadcast = Self;
    type Expand = Self;
    type Reshape = Self;
    type Slice = Self;
    type Transpose = Self;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        match self {
            Self::Dense(this) => this.into_view().broadcast(shape).map(Self::from),
            Self::Sparse(this) => this.into_view().broadcast(shape).map(Self::from),
        }
    }

    fn expand(self, axes: Axes) -> TCResult<Self::Expand> {
        match self {
            Self::Dense(this) => this.into_view().expand(axes).map(Self::from),
            Self::Sparse(this) => this.into_view().expand(axes).map(Self::from),
        }
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        match self {
            Self::Dense(this) => this.into_view().reshape(shape).map(Self::from),
            Self::Sparse(this) => this.into_view().reshape(shape).map(Self::from),
        }
    }

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        match self {
            Self::Dense(this) => this.into_view().slice(range).map(Self::from),
            Self::Sparse(this) => this.into_view().slice(range).map(Self::from),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        match self {
            Self::Dense(this) => this.into_view().transpose(permutation).map(Self::from),
            Self::Sparse(this) => this.into_view().transpose(permutation).map(Self::from),
        }
    }
}

impl<Txn, FE> TensorTrig for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn asin(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().asin().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().asin().map(Self::from),
        }
    }

    fn sin(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().sin().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().sin().map(Self::from),
        }
    }

    fn sinh(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().sinh().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().sinh().map(Self::from),
        }
    }

    fn acos(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().acos().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().acos().map(Self::from),
        }
    }

    fn cos(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().cos().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().cos().map(Self::from),
        }
    }

    fn cosh(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().cosh().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().cosh().map(Self::from),
        }
    }

    fn atan(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().atan().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().atan().map(Self::from),
        }
    }

    fn tan(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().tan().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().tan().map(Self::from),
        }
    }

    fn tanh(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().tanh().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().tanh().map(Self::from),
        }
    }
}

impl<Txn, FE> TensorUnary for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn abs(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().abs().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().abs().map(Self::from),
        }
    }

    fn exp(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().exp().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().exp().map(Self::from),
        }
    }

    fn ln(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().ln().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().ln().map(Self::from),
        }
    }

    fn round(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().round().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().round().map(Self::from),
        }
    }
}

impl<Txn, FE> TensorUnaryBoolean for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Unary = Self;

    fn not(self) -> TCResult<Self::Unary> {
        match self {
            Self::Dense(dense) => dense.into_view().not().map(Self::from),
            Self::Sparse(sparse) => sparse.into_view().not().map(Self::from),
        }
    }
}

#[async_trait]
impl<Txn, FE> TensorWrite for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn write_value(&self, txn_id: TxnId, range: Range, value: Number) -> TCResult<()> {
        match self {
            Self::Dense(dense) => dense.write_value(txn_id, range, value).await,
            Self::Sparse(sparse) => sparse.write_value(txn_id, range, value).await,
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
impl<Txn, FE> TensorWriteDual<Self> for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn write(self, txn_id: TxnId, range: Range, value: Self) -> TCResult<()> {
        match self {
            Self::Dense(this) => this.write(txn_id, range, value.into_dense()).await,
            Self::Sparse(this) => this.write(txn_id, range, value.into_sparse()).await,
        }
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for Tensor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Txn = Txn;
    type View = view::TensorView;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        view::TensorView::read_from(self, *txn.id()).await
    }
}

impl<Txn, FE> From<DenseView<Txn, FE>> for Tensor<Txn, FE> {
    fn from(dense: DenseView<Txn, FE>) -> Self {
        Self::Dense(dense.into())
    }
}

impl<Txn, FE> From<SparseView<Txn, FE>> for Tensor<Txn, FE> {
    fn from(sparse: SparseView<Txn, FE>) -> Self {
        Self::Sparse(sparse.into())
    }
}

impl<Txn, FE> From<Tensor<Txn, FE>> for TensorView<Txn, FE> {
    fn from(tensor: Tensor<Txn, FE>) -> Self {
        match tensor {
            Tensor::Dense(dense) => Self::Dense(dense.into_view()),
            Tensor::Sparse(sparse) => Self::Sparse(sparse.into_view()),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for Tensor<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dense(this) => this.fmt(f),
            Self::Sparse(this) => this.fmt(f),
        }
    }
}

pub enum TensorBase<Txn, FE> {
    Dense(DenseBase<Txn, FE>),
    Sparse(SparseBase<Txn, FE>),
}

impl<Txn, FE> Clone for TensorBase<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Dense(dense) => Self::Dense(dense.clone()),
            Self::Sparse(sparse) => Self::Sparse(sparse.clone()),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> Instance for TensorBase<Txn, FE> {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Dense(dense) => dense.class(),
            Self::Sparse(sparse) => sparse.class(),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> TensorInstance for TensorBase<Txn, FE> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }
}

#[async_trait]
impl<Txn, FE> Transact for TensorBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        match self {
            Self::Dense(dense) => dense.commit(txn_id).await,
            Self::Sparse(sparse) => sparse.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.rollback(txn_id).await,
            Self::Sparse(sparse) => sparse.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.finalize(txn_id).await,
            Self::Sparse(sparse) => sparse.finalize(txn_id).await,
        }
    }
}

#[async_trait]
impl<Txn, FE> fs::Persist<FE> for TensorBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Txn = Txn;
    type Schema = (TensorType, Schema);

    async fn create(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let (class, schema) = schema;
        match class {
            TensorType::Dense => {
                DenseBase::create(txn_id, schema, store)
                    .map_ok(Self::Dense)
                    .await
            }
            TensorType::Sparse => {
                let dtype = schema.dtype;
                let schema = sparse::Schema::new(schema.shape);
                SparseBase::create(txn_id, (dtype, schema), store)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, store: fs::Dir<FE>) -> TCResult<Self> {
        let (class, schema) = schema;
        match class {
            TensorType::Dense => {
                DenseBase::load(txn_id, schema, store)
                    .map_ok(Self::Dense)
                    .await
            }
            TensorType::Sparse => {
                let dtype = schema.dtype;
                let schema = sparse::Schema::new(schema.shape);
                SparseBase::load(txn_id, (dtype, schema), store)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }

    fn dir(&self) -> fs::Inner<FE> {
        match self {
            Self::Dense(dense) => dense.dir(),
            Self::Sparse(sparse) => sparse.dir(),
        }
    }
}

#[async_trait]
impl<Txn, FE> fs::CopyFrom<FE, TensorView<Txn, FE>> for TensorBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn copy_from(
        txn: &Txn,
        store: fs::Dir<FE>,
        instance: TensorView<Txn, FE>,
    ) -> TCResult<Self> {
        match instance {
            TensorView::Dense(dense) => {
                DenseBase::copy_from(txn, store, dense)
                    .map_ok(Self::Dense)
                    .await
            }
            TensorView::Sparse(sparse) => {
                SparseBase::copy_from(txn, store, sparse)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }
}

#[async_trait]
impl<Txn, FE> fs::Restore<FE> for TensorBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn restore(&self, txn_id: TxnId, backup: &Self) -> TCResult<()> {
        match (self, backup) {
            (Self::Dense(this), Self::Dense(that)) => this.restore(txn_id, that).await,
            (Self::Sparse(this), Self::Sparse(that)) => this.restore(txn_id, that).await,
            (this, that) => Err(bad_request!("cannot restore {this:?} from {that:?}")),
        }
    }
}

#[async_trait]
impl<Txn, FE> de::FromStream for TensorBase<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let visitor = TensorVisitor::new(txn);
        decoder.decode_map(visitor).await
    }
}

impl<Txn, FE> From<TensorBase<Txn, FE>> for Tensor<Txn, FE> {
    fn from(base: TensorBase<Txn, FE>) -> Tensor<Txn, FE> {
        match base {
            TensorBase::Dense(base) => Tensor::Dense(Dense::Base(base)),
            TensorBase::Sparse(base) => Tensor::Sparse(Sparse::Base(base)),
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe> fmt::Debug for TensorBase<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dense(dense) => dense.fmt(f),
            Self::Sparse(sparse) => sparse.fmt(f),
        }
    }
}

pub enum TensorView<Txn, FE> {
    Dense(DenseView<Txn, FE>),
    Sparse(SparseView<Txn, FE>),
}

impl<Txn, FE> Clone for TensorView<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::Dense(dense) => Self::Dense(dense.clone()),
            Self::Sparse(sparse) => Self::Sparse(sparse.clone()),
        }
    }
}

struct TensorVisitor<Txn, FE> {
    txn: Txn,
    phantom: PhantomData<FE>,
}

impl<Txn, FE> TensorVisitor<Txn, FE> {
    fn new(txn: Txn) -> Self {
        Self {
            txn,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<Txn, FE> de::Visitor for TensorVisitor<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + Clone,
{
    type Value = TensorBase<Txn, FE>;

    fn expecting() -> &'static str {
        "a tensor"
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let class = map.next_key::<TCPathBuf>(()).await?;
        let class = class.ok_or_else(|| de::Error::invalid_length(0, Self::expecting()))?;
        let class = TensorType::from_path(&class)
            .ok_or_else(|| de::Error::invalid_type(class, "a tensor type (dense or sparse)"))?;

        match class {
            TensorType::Dense => map.next_value(self.txn).map_ok(TensorBase::Dense).await,
            TensorType::Sparse => map.next_value(self.txn).map_ok(TensorBase::Sparse).await,
        }
    }
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
fn size_hint(size: u64) -> usize {
    size.try_into().ok().unwrap_or_else(|| usize::MAX)
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
