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

use tc_btree::Node;
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
    /// A broadcast [`Tensor`]
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
    Sparse,
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
pub enum Tensor<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> {
    Dense(DenseTensor<FD, D, T, DenseAccessor<FD, D, T>>),
    Sparse(FS),
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

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorAccess
    for Tensor<FD, FS, D, T>
{
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ndim(),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Dense(dense) => dense.size(),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorInstance<D>
    for Tensor<FD, FS, D, T>
{
    type Dense = Self;

    fn into_dense(self) -> Self {
        match self {
            Self::Dense(dense) => Self::Dense(dense),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorBoolean<D, Self>
    for Tensor<FD, FS, D, T>
{
    type Combine = Self;

    fn and(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.and(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn or(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.or(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn xor(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(dense) => dense.xor(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

#[async_trait]
impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorCompare<D, Self>
    for Tensor<FD, FS, D, T>
{
    type Compare = Self;
    type Dense = Self;

    async fn eq(self, other: Self, txn: Self::Txn) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.eq(other, txn).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn gt(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.gt(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    async fn gte(self, other: Self, txn: Self::Txn) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.gte(other, txn).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn lt(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.lt(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    async fn lte(self, other: Self, txn: Self::Txn) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.lte(other, txn).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn ne(self, other: Self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.ne(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

#[async_trait]
impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorIO<D>
    for Tensor<FD, FS, D, T>
{
    type Txn = T;

    async fn read_value(&self, txn: &Self::Txn, coord: Coord) -> TCResult<Number> {
        match self {
            Self::Dense(dense) => dense.read_value(txn, coord).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        match self {
            Self::Dense(dense) => dense.write_value(txn_id, bounds, value).await,
            Self::Sparse(_sparse) => todo!(),
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
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

#[async_trait]
impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorDualIO<D, Self>
    for Tensor<FD, FS, D, T>
{
    async fn mask(self, txn: T, other: Self) -> TCResult<()> {
        match self {
            Self::Dense(this) => this.mask(txn, other).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }

    async fn write(self, txn: T, bounds: Bounds, value: Self) -> TCResult<()> {
        debug!("Tensor::write {} to {}", value, bounds);

        match self {
            Self::Dense(this) => this.write(txn, bounds, value).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorMath<D, Self>
    for Tensor<FD, FS, D, T>
{
    type Combine = Self;

    fn add(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.add(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn div(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.div(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn mul(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.mul(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn sub(self, other: Self) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.sub(other),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorReduce<D>
    for Tensor<FD, FS, D, T>
{
    type Reduce = Self;

    fn product(self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.product(axis).map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.product_all(txn),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn sum(self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.sum(axis).map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn sum_all(&self, txn: T) -> TCBoxTryFuture<Number> {
        match self {
            Self::Dense(dense) => dense.sum_all(txn),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorTransform<D>
    for Tensor<FD, FS, D, T>
{
    type Broadcast = Self;
    type Cast = Self;
    type Expand = Self;
    type Slice = Self;
    type Transpose = Self;

    fn cast_into(self, dtype: NumberType) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.cast_into(dtype).map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn broadcast(self, shape: Shape) -> TCResult<Self> {
        if &shape == self.shape() {
            return Ok(self);
        }

        match self {
            Self::Dense(dense) => dense.broadcast(shape).map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn expand_dims(self, axis: usize) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.expand_dims(axis).map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.slice(bounds).map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.transpose(permutation).map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

#[async_trait]
impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> TensorUnary<D>
    for Tensor<FD, FS, D, T>
{
    type Unary = Self;

    fn abs(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.abs().map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }

    async fn all(self, txn: T) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.all(txn).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }

    async fn any(self, txn: T) -> TCResult<bool> {
        match self {
            Self::Dense(dense) => dense.any(txn).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }

    fn not(&self) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.not().map(Self::from),
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

#[async_trait]
impl<'en, FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> Hash<'en, D>
    for Tensor<FD, FS, D, T>
{
    type Item = Array;
    type Txn = T;

    async fn hashable(&'en self, txn: &'en T) -> TCResult<TCTryStream<'en, Self::Item>> {
        match self {
            Self::Dense(dense) => dense.hashable(txn).await,
            Self::Sparse(_sparse) => todo!(),
        }
    }
}

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>, B: DenseAccess<FD, D, T>>
    From<DenseTensor<FD, D, T, B>> for Tensor<FD, FS, D, T>
{
    fn from(dense: DenseTensor<FD, D, T, B>) -> Self {
        Self::Dense(dense.into_inner().accessor().into())
    }
}

#[async_trait]
impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> de::FromStream
    for Tensor<FD, FS, D, T>
where
    <D as Dir>::FileClass: From<TensorType> + Send,
    FD: TryFrom<<D as Dir>::File, Error = TCError>,
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
impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> de::Visitor
    for TensorVisitor<FD, FS, D, T>
where
    <D as Dir>::FileClass: From<TensorType> + Send,
    FD: TryFrom<<D as Dir>::File, Error = TCError>,
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
                map.next_value::<DenseTensor<FD, D, T, BlockListFile<FD, D, T>>>(self.txn)
                    .map_ok(Tensor::from)
                    .await
            }
            TensorType::Sparse => todo!(),
        }
    }
}

#[async_trait]
impl<'en, FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> IntoView<'en, D>
    for Tensor<FD, FS, D, T>
{
    type Txn = T;
    type View = TensorView<'en>;

    async fn into_view(self, txn: T) -> TCResult<Self::View> {
        match self {
            Tensor::Dense(tensor) => tensor.into_view(txn).map_ok(TensorView::Dense).await,
            Tensor::Sparse(_tensor) => todo!(),
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

impl<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>> fmt::Display
    for Tensor<FD, FS, D, T>
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tensor")
    }
}

pub fn broadcast<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>>(
    left: Tensor<FD, FS, D, T>,
    right: Tensor<FD, FS, D, T>,
) -> TCResult<(Tensor<FD, FS, D, T>, Tensor<FD, FS, D, T>)> {
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
