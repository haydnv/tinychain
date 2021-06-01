/// A [`Tensor`], an n-dimensional array of [`Number`]s which supports basic math and logic
use std::convert::TryFrom;
use std::fmt;
use std::iter::FromIterator;
use std::marker::PhantomData;
use std::pin::Pin;

use afarray::Array;
use async_trait::async_trait;
use destream::{de, en};
use futures::{Future, TryFutureExt};
use log::debug;
use number_general::{Number, NumberType};

use tc_error::*;
use tc_transact::fs::{Dir, File, Hash};
use tc_transact::{IntoView, Transaction, TxnId};
use tcgeneric::{
    label, path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCBoxTryFuture,
    TCPathBuf, TCTryStream, Tuple,
};

pub use bounds::{AxisBounds, Bounds, Shape};
pub use dense::{BlockListFile, DenseAccess, DenseAccessor, DenseTensor};

mod bounds;
mod dense;
#[allow(dead_code)]
mod transform;

const PREFIX: PathLabel = path_label(&["state", "collection", "tensor"]);

/// The file extension of a [`Tensor`]
pub const EXT: &str = "array";

type Read<'a> = Pin<Box<dyn Future<Output = TCResult<(Coord, Number)>> + Send + 'a>>;

/// The schema of a [`Tensor`]
pub type Schema = (Shape, NumberType);

/// The address of an individual element in a [`Tensor`].
pub type Coord = Vec<u64>;

/// Trait defining a read operation for a single [`Tensor`] element
pub trait ReadValueAt<D: Dir> {
    /// The transaction context
    type Txn: Transaction<D>;

    /// Read the value of the element at the given [`Coord`].
    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a>;
}

/// Basic properties common to all [`Tensor`]s
pub trait TensorAccess: Send {
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
pub trait TensorInstance<D: Dir>: TensorIO<D> + TensorTransform<D> + Send + Sync {
    /// A dense representation of this [`Tensor`]
    type Dense: TensorInstance<D>;

    /// Return a dense representation of this [`Tensor`].
    fn into_dense(self) -> Self::Dense;
}

/// [`Tensor`] boolean operations.
pub trait TensorBoolean<D: Dir, O>: TensorAccess {
    /// The result type of a boolean operation.
    type Combine: TensorInstance<D>;

    /// Logical and
    fn and(self, other: O) -> TCResult<Self::Combine>;

    /// Logical or
    fn or(self, other: O) -> TCResult<Self::Combine>;

    /// Logical xor
    fn xor(self, other: O) -> TCResult<Self::Combine>;
}

/// Tensor comparison operations
#[async_trait]
pub trait TensorCompare<D: Dir, O>: TensorIO<D> {
    /// The result of a comparison operation
    type Compare: TensorInstance<D>;

    /// The result of a comparison operation which can only return a dense [`Tensor`]
    type Dense: TensorInstance<D>;

    /// Element-wise equality
    async fn eq(self, other: O, txn: Self::Txn) -> TCResult<Self::Dense>;

    /// Element-wise greater-than
    fn gt(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise greater-or-equal
    async fn gte(self, other: O, txn: Self::Txn) -> TCResult<Self::Dense>;

    /// Element-wise less-than
    fn lt(self, other: O) -> TCResult<Self::Compare>;

    /// Element-wise less-or-equal
    async fn lte(self, other: O, txn: Self::Txn) -> TCResult<Self::Dense>;

    /// Element-wise not-equal
    fn ne(self, other: O) -> TCResult<Self::Compare>;
}

/// [`Tensor`] I/O operations
#[async_trait]
pub trait TensorIO<D: Dir>: TensorAccess {
    /// Transaction context type
    type Txn: Transaction<D>;

    /// Read a single value from this [`Tensor`].
    async fn read_value(&self, txn: &Self::Txn, coord: Coord) -> TCResult<Number>;

    /// Write a single value to the slice of this [`Tensor`] with the given [`Bounds`].
    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()>;

    /// Overwrite a single element of this [`Tensor`].
    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

/// [`Tensor`] I/O operations which accept another [`Tensor`] as an argument
#[async_trait]
pub trait TensorDualIO<D: Dir, O>: TensorIO<D> {
    /// Zero out the elements of this [`Tensor`] where the corresponding element of `value` is nonzero.
    async fn mask(self, txn: <Self as TensorIO<D>>::Txn, value: O) -> TCResult<()>;

    /// Overwrite the slice of this [`Tensor`] given by [`Bounds`] with the given `value`.
    async fn write(self, txn: <Self as TensorIO<D>>::Txn, bounds: Bounds, value: O)
        -> TCResult<()>;
}

/// [`Tensor`] math operations
pub trait TensorMath<D: Dir, O>: TensorAccess {
    /// The result type of a math operation
    type Combine: TensorInstance<D>;

    /// Add two tensors together.
    fn add(self, other: O) -> TCResult<Self::Combine>;

    /// Divide `self` by `other`.
    fn div(self, other: O) -> TCResult<Self::Combine>;

    /// Multiply two tensors together.
    fn mul(self, other: O) -> TCResult<Self::Combine>;

    /// Subtract `other` from `self`.
    fn sub(self, other: O) -> TCResult<Self::Combine>;
}

/// [`Tensor`] reduction operations
pub trait TensorReduce<D: Dir>: TensorIO<D> {
    /// The result type of a reduce operation
    type Reduce: TensorInstance<D>;

    /// Return the product of this [`Tensor`] along the given `axis`.
    fn product(self, axis: usize) -> TCResult<Self::Reduce>;

    /// Return the product of all elements in this [`Tensor`].
    fn product_all(&self, txn: <Self as TensorIO<D>>::Txn) -> TCBoxTryFuture<Number>;

    /// Return the sum of this [`Tensor`] along the given `axis`.
    fn sum(self, axis: usize) -> TCResult<Self::Reduce>;

    /// Return the sum of all elements in this [`Tensor`].
    fn sum_all(&self, txn: <Self as TensorIO<D>>::Txn) -> TCBoxTryFuture<Number>;
}

/// [`Tensor`] transforms
pub trait TensorTransform<D: Dir>: TensorAccess {
    /// A broadcasted [`Tensor`]
    type Broadcast: TensorInstance<D>;

    /// A type-cast [`Tensor`]
    type Cast: TensorInstance<D>;

    /// A [`Tensor`] with an expanded dimension
    type Expand: TensorInstance<D>;

    /// A [`Tensor`] slice
    type Slice: TensorInstance<D>;

    /// A transposed [`Tensor`]
    type Transpose: TensorInstance<D>;

    /// Cast this [`Tensor`] to the given `dtype`.
    fn cast_into(self, dtype: NumberType) -> TCResult<Self::Cast>;

    /// Broadcast this [`Tensor`] to the given `shape`.
    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast>;

    /// Insert a new dimension of size 1 at the given `axis`.
    fn expand_dims(self, axis: usize) -> TCResult<Self::Expand>;

    /// Return a slice of this [`Tensor`] with the given `bounds`.
    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    /// Transpose this [`Tensor`] by reordering its axes according to the given `permutation`.
    /// If no permutation is given, the axes will be reversed.
    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose>;
}

/// Unary [`Tensor`] operations
#[async_trait]
pub trait TensorUnary<D: Dir>: TensorIO<D> {
    /// The return type of a unary operation
    type Unary: TensorInstance<D>;

    /// Element-wise absolute value
    fn abs(&self) -> TCResult<Self::Unary>;

    /// Return `true` if all elements in this [`Tensor`] are nonzero.
    async fn all(self, txn: Self::Txn) -> TCResult<bool>;

    /// Return `true` if any element in this [`Tensor`] is nonzero.
    async fn any(self, txn: Self::Txn) -> TCResult<bool>;

    /// Element-wise logical not
    fn not(&self) -> TCResult<Self::Unary>;
}

/// The [`Class`] of [`Tensor`]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum TensorType {
    Dense,
}

impl Class for TensorType {}

impl NativeClass for TensorType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 4 && &path[..3] == &PREFIX[..] {
            match path[3].as_str() {
                "dense" => Some(Self::Dense),
                "sparse" => todo!(),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::Dense => TCPathBuf::from(PREFIX).append(label("dense")),
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("type Tensor")
    }
}

/// An n-dimensional array of numbers which supports basic math and logic operations
#[derive(Clone)]
pub enum Tensor<F: File<Array>, D: Dir, T: Transaction<D>> {
    Dense(DenseTensor<F, D, T, DenseAccessor<F, D, T>>),
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> Instance for Tensor<F, D, T> {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Dense(_) => TensorType::Dense,
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorAccess for Tensor<F, D, T> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Dense(dense) => dense.size(),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorInstance<D> for Tensor<F, D, T> {
    type Dense = Self;

    fn into_dense(self) -> Self {
        match self {
            Self::Dense(dense) => Self::Dense(dense),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorBoolean<D, Self> for Tensor<F, D, T> {
    type Combine = Self;

    fn and(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.and(other),
        }
    }

    fn or(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.or(other),
        }
    }

    fn xor(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.xor(other),
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorCompare<D, Self> for Tensor<F, D, T> {
    type Compare = Self;
    type Dense = Self;

    async fn eq(self, other: Tensor<F, D, T>, txn: Self::Txn) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.eq(other, txn).await,
        }
    }

    fn gt(self, other: Tensor<F, D, T>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.gt(other),
        }
    }

    async fn gte(self, other: Tensor<F, D, T>, txn: Self::Txn) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.gte(other, txn).await,
        }
    }

    fn lt(self, other: Tensor<F, D, T>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.lt(other),
        }
    }

    async fn lte(self, other: Tensor<F, D, T>, txn: Self::Txn) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.lte(other, txn).await,
        }
    }

    fn ne(self, other: Tensor<F, D, T>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.ne(other),
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorIO<D> for Tensor<F, D, T> {
    type Txn = T;

    async fn read_value(&self, txn: &Self::Txn, coord: Coord) -> TCResult<Number> {
        match self {
            Self::Dense(dense) => dense.read_value(txn, coord).await,
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        match self {
            Self::Dense(dense) => dense.write_value(txn_id, bounds, value).await,
        }
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        debug!(
            "Tensor::write_value_at {}, {}",
            Tuple::<u64>::from_iter(coord.to_vec()),
            value
        );

        match self {
            Self::Dense(dense) => dense.write_value_at(txn_id, coord, value).await,
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorDualIO<D, Self> for Tensor<F, D, T> {
    async fn mask(self, txn: T, other: Self) -> TCResult<()> {
        match self {
            Self::Dense(this) => this.mask(txn, other).await,
        }
    }

    async fn write(self, txn: T, bounds: Bounds, value: Self) -> TCResult<()> {
        debug!("Tensor::write {} to {}", value, bounds);

        match self {
            Self::Dense(this) => this.write(txn, bounds, value).await,
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorMath<D, Self> for Tensor<F, D, T> {
    type Combine = Self;

    fn add(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.add(other),
        }
    }

    fn div(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.div(other),
        }
    }

    fn mul(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.mul(other),
        }
    }

    fn sub(self, other: Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.sub(other),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorReduce<D> for Tensor<F, D, T> {
    type Reduce = Self;

    fn product(self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.product(axis).map(Self::from),
        }
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.product_all(txn),
        }
    }

    fn sum(self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.sum(axis).map(Self::from),
        }
    }

    fn sum_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.sum_all(txn),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorTransform<D> for Tensor<F, D, T> {
    type Broadcast = Self;
    type Cast = Self;
    type Expand = Self;
    type Slice = Self;
    type Transpose = Self;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.cast_into(dtype).map(Self::from),
        }
    }

    fn broadcast(self, shape: Shape) -> TCResult<Self> {
        if &shape == self.shape() {
            return Ok(self);
        }

        match self {
            Self::Dense(dense) => dense.broadcast(shape).map(Self::from),
        }
    }

    fn expand_dims(self, axis: usize) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.expand_dims(axis).map(Self::from),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.slice(bounds).map(Self::from),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(Self::from),
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorUnary<D> for Tensor<F, D, T> {
    type Unary = Self;

    fn abs(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.abs().map(Self::from),
        }
    }

    async fn all(self, txn: T) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.all(txn).await,
        }
    }

    async fn any(self, txn: T) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.any(txn).await,
        }
    }

    fn not(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.not().map(Self::from),
        }
    }
}

#[async_trait]
impl<'en, F: File<Array>, D: Dir, T: Transaction<D>> Hash<'en, D> for Tensor<F, D, T> {
    type Item = Array;
    type Txn = T;

    async fn hashable(&'en self, txn: &'en T) -> TCResult<TCTryStream<'en, Self::Item>> {
        match self {
            Self::Dense(dense) => dense.hashable(txn).await,
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    From<DenseTensor<F, D, T, B>> for Tensor<F, D, T>
{
    fn from(dense: DenseTensor<F, D, T, B>) -> Self {
        Self::Dense(dense.into_inner().accessor().into())
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> de::FromStream for Tensor<F, D, T>
where
    <D as Dir>::FileClass: From<TensorType> + Send,
    F: TryFrom<<D as Dir>::File, Error = TCError>,
{
    type Context = T;

    async fn from_stream<De: de::Decoder>(txn: T, decoder: &mut De) -> Result<Self, De::Error> {
        decoder.decode_map(TensorVisitor::new(txn)).await
    }
}

struct TensorVisitor<F, D, T> {
    txn: T,
    dir: PhantomData<D>,
    file: PhantomData<F>,
}

impl<F, D, T> TensorVisitor<F, D, T> {
    fn new(txn: T) -> Self {
        Self {
            txn,
            dir: PhantomData,
            file: PhantomData,
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> de::Visitor for TensorVisitor<F, D, T>
where
    <D as Dir>::FileClass: From<TensorType> + Send,
    F: TryFrom<<D as Dir>::File, Error = TCError>,
{
    type Value = Tensor<F, D, T>;

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
                map.next_value::<DenseTensor<F, D, T, BlockListFile<F, D, T>>>(self.txn)
                    .map_ok(Tensor::from)
                    .await
            }
        }
    }
}

#[async_trait]
impl<'en, F: File<Array>, D: Dir, T: Transaction<D>> IntoView<'en, D> for Tensor<F, D, T> {
    type Txn = T;
    type View = TensorView<'en>;

    async fn into_view(self, txn: T) -> TCResult<Self::View> {
        match self {
            Tensor::Dense(tensor) => tensor.into_view(txn).map_ok(TensorView::Dense).await,
        }
    }
}

/// A view of a [`Tensor`] at a given [`TxnId`], used in serialization
pub enum TensorView<'en> {
    Dense(dense::DenseTensorView<'en>),
}

impl<'en> en::IntoStream<'en> for TensorView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Dense(view) => view.into_stream(encoder),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> fmt::Display for Tensor<F, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tensor")
    }
}

pub fn broadcast<F: File<Array>, D: Dir, T: Transaction<D>>(
    left: Tensor<F, D, T>,
    right: Tensor<F, D, T>,
) -> TCResult<(Tensor<F, D, T>, Tensor<F, D, T>)> {
    if left.shape() == right.shape() {
        return Ok((left, right));
    }

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

    let left_shape = Shape::from(left_shape);
    let right_shape = Shape::from(right_shape);
    let shape = if left_shape.size() > right_shape.size() {
        left_shape
    } else {
        right_shape
    };

    Ok((left.broadcast(shape.clone())?, right.broadcast(shape)?))
}
