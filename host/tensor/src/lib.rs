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

pub const EXT: &str = "array";

type Read<'a> = Pin<Box<dyn Future<Output = TCResult<(Coord, Number)>> + Send + 'a>>;

pub type Schema = (Shape, NumberType);

pub type Coord = Vec<u64>;

pub trait ReadValueAt<D: Dir> {
    type Txn: Transaction<D>;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a>;
}

pub trait TensorAccess: Send {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;
}

pub trait TensorInstance<D: Dir>: TensorIO<D> + TensorTransform<D> + Send + Sync {
    type Dense: TensorInstance<D>;

    fn into_dense(self) -> Self::Dense;
}

#[async_trait]
pub trait TensorIO<D: Dir>: TensorAccess + Sized {
    type Txn: Transaction<D>;

    async fn read_value(&self, txn: &Self::Txn, coord: Coord) -> TCResult<Number>;

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()>;

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

#[async_trait]
pub trait TensorDualIO<D: Dir, O>: TensorIO<D> + Sized {
    async fn mask(&self, txn: <Self as TensorIO<D>>::Txn, value: O) -> TCResult<()>;

    async fn write(
        &self,
        txn: <Self as TensorIO<D>>::Txn,
        bounds: bounds::Bounds,
        value: O,
    ) -> TCResult<()>;
}

pub trait TensorMath<D: Dir, O>: TensorAccess + Sized {
    type Combine: TensorInstance<D>;

    fn add(&self, other: &O) -> TCResult<Self::Combine>;

    fn div(&self, other: &O) -> TCResult<Self::Combine>;

    fn mul(&self, other: &O) -> TCResult<Self::Combine>;

    fn sub(&self, other: &O) -> TCResult<Self::Combine>;
}

pub trait TensorReduce<D: Dir>: TensorIO<D> {
    type Reduce: TensorInstance<D>;

    fn product(&self, axis: usize) -> TCResult<Self::Reduce>;

    fn product_all(&self, txn: <Self as TensorIO<D>>::Txn) -> TCBoxTryFuture<Number>;

    fn sum(&self, axis: usize) -> TCResult<Self::Reduce>;

    fn sum_all(&self, txn: <Self as TensorIO<D>>::Txn) -> TCBoxTryFuture<Number>;
}

pub trait TensorTransform<D: Dir>: TensorAccess {
    type Broadcast: TensorInstance<D>;
    type Cast: TensorInstance<D>;
    type Slice: TensorInstance<D>;

    fn as_type(&self, dtype: NumberType) -> TCResult<Self::Cast>;

    fn broadcast(&self, shape: bounds::Shape) -> TCResult<Self::Broadcast>;

    fn slice(&self, bounds: bounds::Bounds) -> TCResult<Self::Slice>;
}

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
impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorDualIO<D, Tensor<F, D, T>>
    for Tensor<F, D, T>
{
    async fn mask(&self, txn: T, other: Self) -> TCResult<()> {
        match self {
            Self::Dense(this) => this.mask(txn, other).await,
        }
    }

    async fn write(&self, txn: T, bounds: Bounds, value: Self) -> TCResult<()> {
        debug!("Tensor::write {} to {}", value, bounds);

        match self {
            Self::Dense(this) => this.write(txn, bounds, value).await,
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorMath<D, Tensor<F, D, T>> for Tensor<F, D, T> {
    type Combine = Self;

    fn add(&self, other: &Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.add(other),
        }
    }

    fn div(&self, other: &Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.div(other),
        }
    }

    fn mul(&self, other: &Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.mul(other),
        }
    }

    fn sub(&self, other: &Tensor<F, D, T>) -> TCResult<Self::Combine> {
        match self {
            Self::Dense(this) => this.sub(other),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorReduce<D> for Tensor<F, D, T> {
    type Reduce = Self;

    fn product(&self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.product(axis).map(Self::from),
        }
    }

    fn product_all(&self, txn: T) -> TCBoxTryFuture<'_, Number> {
        match self {
            Self::Dense(dense) => dense.product_all(txn),
        }
    }

    fn sum(&self, axis: usize) -> TCResult<Self::Reduce> {
        match self {
            Self::Dense(dense) => dense.sum(axis).map(Self::from),
        }
    }

    fn sum_all(&self, txn: T) -> TCBoxTryFuture<'_, Number> {
        match self {
            Self::Dense(dense) => dense.sum_all(txn),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorTransform<D> for Tensor<F, D, T> {
    type Broadcast = Self;
    type Cast = Self;
    type Slice = Self;

    fn as_type(&self, dtype: NumberType) -> TCResult<Self::Cast> {
        match self {
            Self::Dense(dense) => dense.as_type(dtype).map(Self::from),
        }
    }

    fn broadcast(&self, shape: Shape) -> TCResult<Self::Broadcast> {
        match self {
            Self::Dense(dense) => dense.broadcast(shape).map(Self::from),
        }
    }

    fn slice(&self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::Dense(dense) => dense.slice(bounds).map(Self::from),
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
