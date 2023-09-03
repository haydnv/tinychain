use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use destream::de;
use futures::future::TryFutureExt;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use futures::try_join;
use ha_ndarray::*;
use log::{debug, trace};
use rayon::prelude::*;
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::{Transaction, TxnId};
use tc_value::{DType, Number, NumberType};
use tcgeneric::{TCBoxTryFuture, ThreadSafe};

use crate::tensor::block::Block;
use crate::tensor::dense::{
    DenseAccess, DenseCacheFile, DenseInstance, DenseSlice, DenseTranspose,
};
use crate::tensor::transform::{Broadcast, Expand, Reduce, Reshape, Slice, Transpose};
use crate::tensor::{
    autoqueue, strides_for, Axes, AxisRange, Coord, Range, Semaphore, Shape, TensorInstance,
    TensorPermitRead, TensorPermitWrite,
};

use super::base::SparseBase;
use super::file::{SparseFile, SparseFileWriteGuard};
use super::{stream, unwrap_row, Blocks, Elements, Node, SparseInstance};

#[async_trait]
pub trait SparseWriteLock<'a>: SparseInstance {
    type Guard: SparseWriteGuard<Self::DType>;

    async fn write(&'a self) -> Self::Guard;
}

#[async_trait]
pub trait SparseWriteGuard<T: CDatatype + DType>: Send + Sync {
    async fn clear(&mut self, txn_id: TxnId, range: Range) -> TCResult<()>;

    async fn merge<FE>(
        &mut self,
        txn_id: TxnId,
        filled: SparseFile<FE, T>,
        zeros: SparseFile<FE, T>,
    ) -> TCResult<()>
    where
        FE: AsType<Node> + ThreadSafe,
        Number: CastInto<T>,
    {
        let mut zeros = {
            let table = zeros.into_table().read().await;
            table.into_rows()
        };

        while let Some(row) = zeros.try_next().await? {
            let (coord, zero) = unwrap_row(row);
            self.write_value(txn_id, coord, zero).await?;
        }

        let mut filled = {
            let table = filled.into_table().read().await;
            table.into_rows()
        };

        while let Some(row) = filled.try_next().await? {
            let (coord, value) = unwrap_row(row);
            self.write_value(txn_id, coord, value).await?;
        }

        Ok(())
    }

    async fn overwrite<O>(&mut self, txn_id: TxnId, other: O) -> TCResult<()>
    where
        O: SparseInstance<DType = T> + TensorPermitRead;

    async fn write_value(&mut self, txn_id: TxnId, coord: Coord, value: T) -> TCResult<()>;
}

pub enum SparseAccessCast<Txn, FE> {
    F32(SparseAccess<Txn, FE, f32>),
    F64(SparseAccess<Txn, FE, f64>),
    I16(SparseAccess<Txn, FE, i16>),
    I32(SparseAccess<Txn, FE, i32>),
    I64(SparseAccess<Txn, FE, i64>),
    U8(SparseAccess<Txn, FE, u8>),
    U16(SparseAccess<Txn, FE, u16>),
    U32(SparseAccess<Txn, FE, u32>),
    U64(SparseAccess<Txn, FE, u64>),
}

impl<Txn, FE> Clone for SparseAccessCast<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::F32(access) => Self::F32(access.clone()),
            Self::F64(access) => Self::F64(access.clone()),
            Self::I16(access) => Self::I16(access.clone()),
            Self::I32(access) => Self::I32(access.clone()),
            Self::I64(access) => Self::I64(access.clone()),
            Self::U8(access) => Self::U8(access.clone()),
            Self::U16(access) => Self::U16(access.clone()),
            Self::U32(access) => Self::U32(access.clone()),
            Self::U64(access) => Self::U64(access.clone()),
        }
    }
}

macro_rules! access_cast_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            SparseAccessCast::F32($var) => $call,
            SparseAccessCast::F64($var) => $call,
            SparseAccessCast::I16($var) => $call,
            SparseAccessCast::I32($var) => $call,
            SparseAccessCast::I64($var) => $call,
            SparseAccessCast::U8($var) => $call,
            SparseAccessCast::U16($var) => $call,
            SparseAccessCast::U32($var) => $call,
            SparseAccessCast::U64($var) => $call,
        }
    };
}

macro_rules! access_cast_dispatch_dual {
    ($self:ident, $other:ident, $this:ident, $that:ident, $call:expr) => {
        match ($self, $other) {
            (Self::F32($this), Self::F32($that)) => $call,
            (Self::F64($this), Self::F64($that)) => $call,
            (Self::I16($this), Self::I16($that)) => $call,
            (Self::I32($this), Self::I32($that)) => $call,
            (Self::I64($this), Self::I64($that)) => $call,
            (Self::U8($this), Self::U8($that)) => $call,
            (Self::U16($this), Self::U16($that)) => $call,
            (Self::U32($this), Self::U32($that)) => $call,
            (Self::U64($this), Self::U64($that)) => $call,
            ($this, $that) => Err(bad_request!("cannot merge {:?} and {:?}", $this, $that)),
        }
    };
}

impl<Txn: ThreadSafe, FE: ThreadSafe> SparseAccessCast<Txn, FE> {
    pub fn dtype(&self) -> NumberType {
        access_cast_dispatch!(self, this, this.dtype())
    }

    pub fn shape(&self) -> &Shape {
        access_cast_dispatch!(self, this, this.shape())
    }
}

impl<Txn, FE> SparseAccessCast<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn merge_blocks_inner(
        self,
        other: Self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> TCResult<Pin<Box<dyn Stream<Item = TCResult<(Array<u64>, (Block, Block))>> + Send>>> {
        let shape = if self.shape() == other.shape() {
            Ok(self.shape().clone())
        } else {
            Err(bad_request!("cannot merge {:?} with {:?}", self, other))
        }?;

        access_cast_dispatch_dual!(self, other, this, that, {
            let blocks = merge_blocks_inner(this, that, txn_id, shape, range, order).await?;
            let blocks = blocks.map_ok(|(coords, (left, right))| {
                (
                    coords,
                    (
                        Block::from(Array::from(left)),
                        Block::from(Array::from(right)),
                    ),
                )
            });

            Ok(Box::pin(blocks))
        })
    }

    async fn merge_blocks_outer(
        self,
        other: Self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> TCResult<Pin<Box<dyn Stream<Item = TCResult<(Array<u64>, (Block, Block))>> + Send>>> {
        let shape = if self.shape() == other.shape() {
            Ok(self.shape().clone())
        } else {
            Err(bad_request!("cannot merge {:?} with {:?}", self, other))
        }?;

        access_cast_dispatch_dual!(self, other, this, that, {
            let blocks = merge_blocks_outer(this, that, txn_id, shape, range, order).await?;
            let blocks = blocks.map_ok(|(coords, (left, right))| {
                (
                    coords,
                    (
                        Block::from(Array::from(left)),
                        Block::from(Array::from(right)),
                    ),
                )
            });

            Ok(Box::pin(blocks))
        })
    }

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> TCResult<Blocks<Array<u64>, Block>> {
        access_cast_dispatch!(self, this, {
            let blocks = this.blocks(txn_id, range, order).await?;
            let blocks = blocks.map_ok(|(coords, values)| (coords.into(), values.into()));
            Ok(Box::pin(blocks))
        })
    }

    pub async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> TCResult<Elements<Number>> {
        access_cast_dispatch!(self, this, {
            let elements = this.elements(txn_id, range, order).await?;

            Ok(Box::pin(
                elements.map_ok(|(coord, value)| (coord, Number::from(value))),
            ))
        })
    }

    pub async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        access_cast_dispatch!(
            self,
            this,
            this.read_value(txn_id, coord).map_ok(Number::from).await
        )
    }
}

#[async_trait]
impl<Txn: ThreadSafe, FE: ThreadSafe> TensorPermitRead for SparseAccessCast<Txn, FE> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        access_cast_dispatch!(self, this, this.read_permit(txn_id, range).await)
    }
}

macro_rules! access_cast_from {
    ($t:ty, $var:ident) => {
        impl<Txn, FE> From<SparseAccess<Txn, FE, $t>> for SparseAccessCast<Txn, FE> {
            fn from(access: SparseAccess<Txn, FE, $t>) -> Self {
                Self::$var(access)
            }
        }
    };
}

access_cast_from!(f32, F32);
access_cast_from!(f64, F64);
access_cast_from!(i16, I16);
access_cast_from!(i32, I32);
access_cast_from!(i64, I64);
access_cast_from!(u8, U8);
access_cast_from!(u16, U16);
access_cast_from!(u32, U32);
access_cast_from!(u64, U64);

impl<Txn, FE> fmt::Debug for SparseAccessCast<Txn, FE>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        access_cast_dispatch!(self, this, this.fmt(f))
    }
}

pub enum SparseAccess<Txn, FE, T: CDatatype> {
    Base(SparseBase<Txn, FE, T>),
    Table(SparseFile<FE, T>),
    Broadcast(Box<SparseBroadcast<Txn, FE, T>>),
    BroadcastAxis(Box<SparseBroadcastAxis<Self>>),
    Combine(Box<SparseCombine<Self, Self, T>>),
    CombineLeft(Box<SparseCombineLeft<Self, Self, T>>),
    CombineConst(Box<SparseCombineConst<Self, T>>),
    Compare(Box<SparseCompare<Txn, FE, T>>),
    CompareConst(Box<SparseCompareConst<Txn, FE, T>>),
    CompareLeft(Box<SparseCompareLeft<Txn, FE, T>>),
    Cond(Box<SparseCond<SparseAccess<Txn, FE, u8>, Self, Self>>),
    Cow(Box<SparseCow<FE, T, Self>>),
    Dense(Box<SparseDense<Txn, FE, T>>),
    Expand(Box<SparseExpand<Self>>),
    Reduce(Box<SparseReduce<Txn, FE, T>>),
    Reshape(Box<SparseReshape<Self>>),
    Slice(Box<SparseSlice<Self>>),
    Transpose(Box<SparseTranspose<Self>>),
    Unary(Box<SparseUnary<Self, T>>),
    UnaryCast(Box<SparseUnaryCast<Txn, FE, T>>),
    Version(SparseVersion<FE, T>),
}

impl<Txn, FE, T: CDatatype> Clone for SparseAccess<Txn, FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Base(base) => Self::Base(base.clone()),
            Self::Table(table) => Self::Table(table.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::BroadcastAxis(broadcast) => Self::BroadcastAxis(broadcast.clone()),
            Self::Combine(combine) => Self::Combine(combine.clone()),
            Self::CombineLeft(combine) => Self::CombineLeft(combine.clone()),
            Self::CombineConst(combine) => Self::CombineConst(combine.clone()),
            Self::Compare(compare) => Self::Compare(compare.clone()),
            Self::CompareConst(compare) => Self::CompareConst(compare.clone()),
            Self::CompareLeft(compare) => Self::CompareLeft(compare.clone()),
            Self::Cond(cond) => Self::Cond(cond.clone()),
            Self::Cow(cow) => Self::Cow(cow.clone()),
            Self::Dense(dense) => Self::Dense(dense.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reduce(reduce) => Self::Reduce(reduce.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
            Self::Unary(unary) => Self::Unary(unary.clone()),
            Self::UnaryCast(unary) => Self::UnaryCast(unary.clone()),
            Self::Version(version) => Self::Version(version.clone()),
        }
    }
}

macro_rules! access_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Base($var) => $call,
            Self::Table($var) => $call,
            Self::Broadcast($var) => $call,
            Self::BroadcastAxis($var) => $call,
            Self::Combine($var) => $call,
            Self::CombineLeft($var) => $call,
            Self::CombineConst($var) => $call,
            Self::Compare($var) => $call,
            Self::CompareConst($var) => $call,
            Self::CompareLeft($var) => $call,
            Self::Cond($var) => $call,
            Self::Cow($var) => $call,
            Self::Dense($var) => $call,
            Self::Expand($var) => $call,
            Self::Reduce($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
            Self::Unary($var) => $call,
            Self::UnaryCast($var) => $call,
            Self::Version($var) => $call,
            Self::Transpose($var) => $call,
        }
    };
}

impl<Txn, FE, T: CDatatype> TensorInstance for SparseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        access_dispatch!(self, this, this.dtype())
    }

    fn shape(&self) -> &Shape {
        access_dispatch!(self, this, this.shape())
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseAccess<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + AsType<Buffer<T>>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Array<u64>, Array<T>>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        access_dispatch!(self, this, {
            let blocks = this.blocks(txn_id, range, order).await?;

            let blocks =
                blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

            Ok(Box::pin(blocks))
        })
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        access_dispatch!(self, this, this.elements(txn_id, range, order).await)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        access_dispatch!(self, this, this.read_value(txn_id, coord).await)
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        match self {
            Self::Base(base) => base.read_permit(txn_id, range).await,
            Self::Broadcast(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::BroadcastAxis(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::Cow(cow) => cow.read_permit(txn_id, range).await,
            Self::Combine(combine) => combine.read_permit(txn_id, range).await,
            Self::CombineLeft(combine) => combine.read_permit(txn_id, range).await,
            Self::CombineConst(combine) => combine.read_permit(txn_id, range).await,
            Self::Compare(compare) => compare.read_permit(txn_id, range).await,
            Self::CompareLeft(compare) => compare.read_permit(txn_id, range).await,
            Self::Cond(cond) => cond.read_permit(txn_id, range).await,
            Self::Dense(dense) => dense.read_permit(txn_id, range).await,
            Self::Expand(expand) => expand.read_permit(txn_id, range).await,
            Self::Reduce(reduce) => reduce.read_permit(txn_id, range).await,
            Self::Reshape(reshape) => reshape.read_permit(txn_id, range).await,
            Self::Slice(slice) => slice.read_permit(txn_id, range).await,
            Self::Transpose(transpose) => transpose.read_permit(txn_id, range).await,
            Self::Unary(unary) => unary.read_permit(txn_id, range).await,
            Self::UnaryCast(unary) => unary.read_permit(txn_id, range).await,
            Self::Version(version) => version.read_permit(txn_id, range).await,
            other => Err(bad_request!(
                "{:?} does not support transactional reads",
                other
            )),
        }
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitWrite for SparseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        match self {
            Self::Base(base) => base.write_permit(txn_id, range).await,
            Self::Slice(slice) => slice.write_permit(txn_id, range).await,
            Self::Version(version) => version.write_permit(txn_id, range).await,
            other => Err(bad_request!(
                "{:?} does not support transactional writes",
                other
            )),
        }
    }
}

impl<Txn, FE, T> fmt::Debug for SparseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        access_dispatch!(self, this, this.fmt(f))
    }
}

pub struct SparseBroadcast<Txn, FE, T: CDatatype> {
    transform: Broadcast,
    source: SparseAccess<Txn, FE, T>,
}

impl<Txn, FE, T: CDatatype> Clone for SparseBroadcast<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            transform: self.transform.clone(),
            source: self.source.clone(),
        }
    }
}

impl<Txn, FE, T> SparseBroadcast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    pub fn new<S>(source: S, shape: Shape) -> TCResult<Self>
    where
        S: TensorInstance + Into<SparseAccess<Txn, FE, T>> + fmt::Debug,
    {
        log::debug!("SparseBroadcast::new {:?} into {:?}", source.shape(), shape);

        let offset = if shape.len() >= source.ndim() {
            Ok(shape.len() - source.ndim())
        } else {
            Err(bad_request!("cannot broadcast {source:?} into {shape:?}"))
        }?;

        let source = if offset == 0 {
            Ok(source.into())
        } else {
            SparseExpand::new(source, vec![0; offset]).map(SparseAccess::from)
        }?;

        Broadcast::new(source.shape().clone(), shape).map(|transform| Self {
            transform,
            source: source.into(),
        })
    }
}

impl<Txn, FE, T> TensorInstance for SparseBroadcast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseBroadcast<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node> + AsType<Buffer<T>>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(txn_id, range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        log::debug!("SparseBroadcast::elements in range {range:?} with order {order:?}");

        self.shape().validate_range(&range)?;
        self.shape().validate_axes(&order, true)?;

        let (source, range, shape) = if order.iter().copied().enumerate().all(|(i, x)| i == x) {
            (self.source, range, self.transform.shape().clone())
        } else {
            let range = order
                .iter()
                .copied()
                .map(|x| {
                    range
                        .get(x)
                        .cloned()
                        .unwrap_or_else(|| AxisRange::all(self.shape()[x]))
                })
                .collect();

            let shape = order.iter().copied().map(|x| self.shape()[x]).collect();

            let source = SparseTranspose::new(self.source, Some(order)).map(SparseAccess::from)?;

            (source, range, shape)
        };

        let dims = source.shape().to_vec().into_iter().zip(shape);

        let mut inner = source;

        for (x, (dim, bdim)) in dims.enumerate().rev() {
            if dim == bdim {
                // no-op
            } else if dim == 1 {
                let broadcast_axis = SparseBroadcastAxis::new(inner, x, bdim)?;
                inner = SparseAccess::BroadcastAxis(Box::new(broadcast_axis));
            } else {
                return Err(bad_request!(
                    "cannot broadcast {} into {} at axis {}",
                    dim,
                    bdim,
                    x
                ));
            }
        }

        inner.elements(txn_id, range, Axes::default()).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, source_coord).await
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseBroadcast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.shape().validate_range(&range)?;
        let source_range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, source_range).await
    }
}

impl<Txn, FE, T: CDatatype> From<SparseBroadcast<Txn, FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(accessor: SparseBroadcast<Txn, FE, T>) -> Self {
        Self::Broadcast(Box::new(accessor))
    }
}

impl<Txn, FE, T: CDatatype> fmt::Debug for SparseBroadcast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "broadcast {:?} into {:?}",
            self.source,
            self.transform.shape()
        )
    }
}

#[derive(Clone)]
pub struct SparseBroadcastAxis<S> {
    source: S,
    axis: usize,
    dim: u64,
    shape: Shape,
}

impl<S: TensorInstance + fmt::Debug> SparseBroadcastAxis<S> {
    fn new(source: S, axis: usize, dim: u64) -> TCResult<Self> {
        log::debug!("SparseBroadcastAxis::new: broadcast axis {axis} of {source:?} into {dim}");

        let shape = if axis < source.ndim() {
            let mut shape = source.shape().to_vec();
            if shape[axis] == 1 {
                shape[axis] = dim;
                Ok(shape)
            } else {
                Err(bad_request!(
                    "cannot broadcast dimension {} into {}",
                    shape[axis],
                    dim
                ))
            }
        } else {
            Err(bad_request!("invalid axis for {:?}: {}", source, axis))
        }?;

        Ok(Self {
            source,
            axis,
            dim,
            shape: shape.into(),
        })
    }
}

impl<S: TensorInstance> TensorInstance for SparseBroadcastAxis<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance + Clone> SparseInstance for SparseBroadcastAxis<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(txn_id, range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        self.shape.validate_range(&range)?;
        self.shape.validate_axes(&order, true)?;

        let order = if order.is_empty() {
            (0..self.ndim()).into_iter().collect()
        } else if order.iter().copied().enumerate().all(|(i, x)| i == x) {
            order
        } else {
            return Err(not_implemented!("transpose a broadcasted sparse tensor"));
        };

        let ndim = self.shape.len();

        let axis_range = range
            .get(self.axis)
            .cloned()
            .unwrap_or_else(|| AxisRange::all(self.shape()[self.axis]));

        let (source_range, dim) = if range.len() > self.axis {
            let bdim = range[self.axis].dim();
            let mut source_range = range;
            source_range[self.axis] = AxisRange::At(0);
            (source_range, bdim)
        } else {
            (range, self.dim)
        };

        if self.axis == 0 {
            let source = self.source;
            let elements = futures::stream::iter(axis_range)
                .map(move |outer_i| {
                    let source = source.clone();
                    let source_range = source_range.clone();
                    let source_order = order.to_vec();

                    async move {
                        let source_elements =
                            source.elements(txn_id, source_range, source_order).await?;

                        let elements = source_elements.map_ok(move |(mut inner_coord, value)| {
                            debug_assert_eq!(inner_coord.len(), ndim);
                            inner_coord[0] = outer_i;
                            (inner_coord, value)
                        });

                        TCResult::Ok(elements)
                    }
                })
                .buffered(num_cpus::get())
                .try_flatten();

            Ok(Box::pin(elements))
        } else if self.axis == self.ndim() - 1 {
            let source_elements = self.source.elements(txn_id, source_range, order).await?;

            // TODO: write a range to a slice of a coordinate block instead
            let elements = source_elements
                .map_ok(move |(source_coord, value)| {
                    futures::stream::iter(axis_range.clone()).map(move |i| {
                        debug_assert_eq!(source_coord.len(), ndim);
                        let mut coord = source_coord.to_vec();
                        *coord.last_mut().expect("x") = i;
                        Ok((coord, value))
                    })
                })
                .try_flatten();

            Ok(Box::pin(elements))
        } else {
            let inner_range = source_range
                .iter()
                .skip(self.axis)
                .cloned()
                .collect::<Vec<_>>();

            let filled = self
                .source
                .clone()
                .filled_at(
                    txn_id,
                    source_range,
                    Axes::default(),
                    (0..self.axis).into_iter().collect(),
                )
                .await?;

            let elements = filled
                .map(move |result| {
                    let outer_coord = result?;
                    debug_assert_eq!(outer_coord.len(), self.axis);

                    let inner_range = inner_range.to_vec();

                    let prefix = outer_coord
                        .iter()
                        .copied()
                        .map(|i| AxisRange::At(i))
                        .collect();

                    log::trace!("broadcast slice at {prefix:?} x{dim}");

                    let slice = SparseSlice::new(self.source.clone(), prefix)?;

                    let elements = futures::stream::iter(axis_range.clone())
                        .map(move |i| {
                            let outer_coord = outer_coord.to_vec();
                            let inner_range = inner_range.to_vec().into();
                            let slice = slice.clone();

                            async move {
                                trace!("stream over elements in slice {inner_range:?}");

                                let inner_elements =
                                    slice.elements(txn_id, inner_range, Axes::default()).await?;

                                let elements =
                                    inner_elements.map_ok(move |(inner_coord, value)| {
                                        log::trace!("outer coord is {outer_coord:?}, i is {i}, inner coord is {inner_coord:?}, ndim is {ndim}");

                                        let mut coord = Coord::with_capacity(ndim);
                                        coord.extend(outer_coord.iter().copied());
                                        coord.extend(inner_coord);

                                        coord[self.axis] = i;

                                        debug_assert_eq!(coord.len(), ndim);

                                        (coord, value)
                                    });

                                Result::<_, TCError>::Ok(elements)
                            }
                        })
                        .buffered(num_cpus::get())
                        .try_flatten();

                    Result::<_, TCError>::Ok(elements)
                })
                .try_flatten();

            Ok(Box::pin(elements))
        }
    }

    async fn read_value(&self, txn_id: TxnId, mut coord: Coord) -> Result<Self::DType, TCError> {
        self.shape.validate_coord(&coord)?;
        coord[self.axis] = 0;
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for SparseBroadcastAxis<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        mut range: Range,
    ) -> TCResult<Vec<PermitRead<Range>>> {
        self.shape.validate_range(&range)?;

        if range.len() > self.axis {
            range[self.axis] = AxisRange::At(0);
        }

        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T, S> From<SparseBroadcastAxis<S>> for SparseAccess<Txn, FE, T>
where
    T: CDatatype,
    S: Into<SparseAccess<Txn, FE, T>>,
{
    fn from(broadcast: SparseBroadcastAxis<S>) -> Self {
        Self::BroadcastAxis(Box::new(SparseBroadcastAxis {
            source: broadcast.source.into(),
            axis: broadcast.axis,
            dim: broadcast.dim,
            shape: broadcast.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseBroadcastAxis<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "broadcast of {:?} axis {}", self.source, self.axis)
    }
}

#[derive(Clone)]
pub struct SparseCombine<L, R, T: CDatatype> {
    left: L,
    right: R,
    block_op: fn(Array<T>, Array<T>) -> TCResult<Array<T>>,
    value_op: fn(T, T) -> T,
}

impl<L, R, T> SparseCombine<L, R, T>
where
    L: SparseInstance<DType = T> + fmt::Debug,
    R: SparseInstance<DType = T> + fmt::Debug,
    T: CDatatype + DType,
{
    pub fn new(
        left: L,
        right: R,
        block_op: fn(Array<T>, Array<T>) -> TCResult<Array<T>>,
        value_op: fn(T, T) -> T,
    ) -> TCResult<Self> {
        log::debug!("SparseCombine::new({left:?}, {right:?})");

        if left.shape() == right.shape() {
            Ok(Self {
                left,
                right,
                block_op,
                value_op,
            })
        } else {
            Err(bad_request!(
                "cannot combine {:?} and {:?} (wrong shape)",
                left,
                right
            ))
        }
    }
}

impl<L, R, T> TensorInstance for SparseCombine<L, R, T>
where
    L: TensorInstance,
    R: TensorInstance,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.left.shape(), self.right.shape());
        self.left.shape()
    }
}

#[async_trait]
impl<L, R, T> SparseInstance for SparseCombine<L, R, T>
where
    L: SparseInstance<DType = T>,
    R: SparseInstance<DType = T>,
    T: CDatatype + DType + PartialEq + fmt::Debug,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();

        let block_op = self.block_op;
        let shape = self.shape().clone();
        let source_blocks =
            merge_blocks_outer(self.left, self.right, txn_id, shape, range, order).await?;

        let blocks = source_blocks
            .map(move |result| {
                result.and_then(|(coords, (left, right))| {
                    debug_assert_eq!(coords.ndim(), 2);
                    debug_assert_eq!(coords.shape()[1], ndim);
                    (block_op)(left.into(), right.into()).map(|values| (coords, values))
                })
            })
            .try_filter_map(
                move |(coords, values)| async move { filter_zeros(coords, values, ndim) },
            );

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        let (left, right) = try_join!(
            self.left.read_value(txn_id, coord.to_vec()),
            self.right.read_value(txn_id, coord)
        )?;

        Ok((self.value_op)(left, right))
    }
}

#[async_trait]
impl<L, R, T> TensorPermitRead for SparseCombine<L, R, T>
where
    L: TensorPermitRead,
    R: TensorPermitRead,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these locks in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, L, R, T> From<SparseCombine<L, R, T>> for SparseAccess<Txn, FE, T>
where
    L: Into<SparseAccess<Txn, FE, T>>,
    R: Into<SparseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    fn from(combine: SparseCombine<L, R, T>) -> Self {
        Self::Combine(Box::new(SparseCombine {
            left: combine.left.into(),
            right: combine.right.into(),
            block_op: combine.block_op,
            value_op: combine.value_op,
        }))
    }
}

impl<L: fmt::Debug, R: fmt::Debug, T: CDatatype + DType> fmt::Debug for SparseCombine<L, R, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "outer join of {:?} and {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct SparseCombineLeft<L, R, T: CDatatype> {
    left: L,
    right: R,
    block_op: fn(Array<T>, Array<T>) -> TCResult<Array<T>>,
    value_op: fn(T, T) -> T,
}

impl<L, R, T> SparseCombineLeft<L, R, T>
where
    L: SparseInstance<DType = T> + fmt::Debug,
    R: SparseInstance<DType = T> + fmt::Debug,
    T: CDatatype + DType,
{
    pub fn new(
        left: L,
        right: R,
        block_op: fn(Array<T>, Array<T>) -> TCResult<Array<T>>,
        value_op: fn(T, T) -> T,
    ) -> TCResult<Self> {
        if left.shape() == right.shape() {
            log::debug!("SparseCombineLeft::new({left:?}, {right:?})");

            Ok(Self {
                left,
                right,
                block_op,
                value_op,
            })
        } else {
            Err(bad_request!(
                "cannot combine {left:?} and {right:?} (wrong shape)"
            ))
        }
    }
}

impl<L, R, T> TensorInstance for SparseCombineLeft<L, R, T>
where
    L: TensorInstance,
    R: TensorInstance,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.left.shape(), self.right.shape());
        self.left.shape()
    }
}

#[async_trait]
impl<L, R, T> SparseInstance for SparseCombineLeft<L, R, T>
where
    L: SparseInstance<DType = T>,
    R: SparseInstance<DType = T>,
    T: CDatatype + DType + PartialEq + fmt::Debug,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();

        let block_op = self.block_op;
        let shape = self.shape().clone();
        let source_blocks =
            merge_blocks_inner(self.left, self.right, txn_id, shape, range, order).await?;

        let blocks = source_blocks
            .map(move |result| {
                result.and_then(|(coords, (left, right))| {
                    debug_assert_eq!(coords.ndim(), 2);
                    debug_assert_eq!(coords.shape()[1], ndim);
                    trace!("combine values {left:?} and {right:?} at {coords:?}");
                    (block_op)(left.into(), right.into()).map(|values| (coords, values))
                })
            })
            .try_filter_map(
                move |(coords, values)| async move { filter_zeros(coords, values, ndim) },
            );

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        let (left, right) = try_join!(
            self.left.read_value(txn_id, coord.to_vec()),
            self.right.read_value(txn_id, coord)
        )?;

        Ok((self.value_op)(left, right))
    }
}

#[async_trait]
impl<L, R, T> TensorPermitRead for SparseCombineLeft<L, R, T>
where
    L: TensorPermitRead,
    R: TensorPermitRead,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these locks in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, L, R, T> From<SparseCombineLeft<L, R, T>> for SparseAccess<Txn, FE, T>
where
    FE: ThreadSafe,
    L: Into<SparseAccess<Txn, FE, T>>,
    R: Into<SparseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    fn from(combine: SparseCombineLeft<L, R, T>) -> Self {
        Self::CombineLeft(Box::new(SparseCombineLeft {
            left: combine.left.into(),
            right: combine.right.into(),
            block_op: combine.block_op,
            value_op: combine.value_op,
        }))
    }
}

impl<L: fmt::Debug, R: fmt::Debug, T: CDatatype + DType> fmt::Debug for SparseCombineLeft<L, R, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "inner join (left combine) of {:?} and {:?}",
            self.left, self.right
        )
    }
}

#[derive(Clone)]
pub struct SparseCombineConst<S, T: CDatatype> {
    left: S,
    right: Number,
    block_op: fn(Array<T>, Number) -> TCResult<Array<T>>,
    value_op: fn(T, Number) -> T,
}

impl<S, T: CDatatype> SparseCombineConst<S, T> {
    pub fn new(
        left: S,
        right: Number,
        block_op: fn(Array<T>, Number) -> TCResult<Array<T>>,
        value_op: fn(T, Number) -> T,
    ) -> Self {
        Self {
            left,
            right,
            block_op,
            value_op,
        }
    }
}

impl<S, T> TensorInstance for SparseCombineConst<S, T>
where
    S: TensorInstance,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.left.shape()
    }
}

#[async_trait]
impl<S, T> SparseInstance for SparseCombineConst<S, T>
where
    S: SparseInstance<DType = T>,
    T: CDatatype + DType,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();

        let left_blocks = self.left.blocks(txn_id, range, order).await?;

        let blocks = left_blocks
            .map(move |result| {
                let (coords, values) = result?;
                let values = (self.block_op)(values.into(), self.right)?;
                Ok((coords, values))
            })
            .try_filter_map(move |(coords, values)| async move {
                filter_zeros(coords.into(), values, ndim)
            });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.left
            .read_value(txn_id, coord)
            .map_ok(|value| (self.value_op)(value, self.right))
            .await
    }
}

#[async_trait]
impl<S, T> TensorPermitRead for SparseCombineConst<S, T>
where
    S: TensorPermitRead,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<SparseCombineConst<S, T>> for SparseAccess<Txn, FE, T>
where
    S: Into<SparseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    fn from(combine: SparseCombineConst<S, T>) -> Self {
        Self::CombineConst(Box::new(SparseCombineConst {
            left: combine.left.into(),
            right: combine.right,
            block_op: combine.block_op,
            value_op: combine.value_op,
        }))
    }
}

impl<S, T> fmt::Debug for SparseCombineConst<S, T>
where
    S: fmt::Debug,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with a constant value", self.left)
    }
}

pub struct SparseCompare<Txn, FE, T: CDatatype> {
    left: SparseAccessCast<Txn, FE>,
    right: SparseAccessCast<Txn, FE>,
    block_op: fn(Block, Block) -> TCResult<Array<T>>,
    value_op: fn(Number, Number) -> T,
}

impl<Txn, FE, T: CDatatype> Clone for SparseCompare<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe, T: CDatatype> SparseCompare<Txn, FE, T> {
    pub fn new<L, R>(
        left: L,
        right: R,
        block_op: fn(Block, Block) -> TCResult<Array<T>>,
        value_op: fn(Number, Number) -> T,
    ) -> TCResult<Self>
    where
        L: Into<SparseAccessCast<Txn, FE>>,
        R: Into<SparseAccessCast<Txn, FE>>,
    {
        let left = left.into();
        let right = right.into();

        if left.shape() == right.shape() {
            Ok(Self {
                left,
                right,
                block_op,
                value_op,
            })
        } else {
            Err(bad_request!(
                "cannot compare {:?} and {:?} (wrong shape)",
                left,
                right
            ))
        }
    }
}

impl<Txn, FE, T> TensorInstance for SparseCompare<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.left.shape(), self.right.shape());
        self.left.shape()
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseCompare<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
    T: CDatatype + DType,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();

        let source_blocks = self
            .left
            .merge_blocks_outer(self.right, txn_id, range, order)
            .await?;

        let blocks = source_blocks
            .map(move |result| {
                let (coords, (left, right)) = result?;
                let values = (self.block_op)(left, right)?;
                Ok((coords, values))
            })
            .try_filter_map(
                move |(coords, values)| async move { filter_zeros(coords, values, ndim) },
            );

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        let (left, right) = try_join!(
            self.left.read_value(txn_id, coord.to_vec()),
            self.right.read_value(txn_id, coord)
        )?;

        Ok((self.value_op)(left, right))
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseCompare<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, T: CDatatype> From<SparseCompare<Txn, FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(compare: SparseCompare<Txn, FE, T>) -> Self {
        Self::Compare(Box::new(compare))
    }
}

impl<Txn, FE, T> fmt::Debug for SparseCompare<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} and {:?}", self.left, self.right)
    }
}

pub struct SparseCompareLeft<Txn, FE, T: CDatatype> {
    left: SparseAccessCast<Txn, FE>,
    right: SparseAccessCast<Txn, FE>,
    block_op: fn(Block, Block) -> TCResult<Array<T>>,
    value_op: fn(Number, Number) -> T,
}

impl<Txn, FE, T: CDatatype> Clone for SparseCompareLeft<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn: ThreadSafe, FE: ThreadSafe, T: CDatatype> SparseCompareLeft<Txn, FE, T> {
    pub fn new<L, R>(
        left: L,
        right: R,
        block_op: fn(Block, Block) -> TCResult<Array<T>>,
        value_op: fn(Number, Number) -> T,
    ) -> TCResult<Self>
    where
        L: Into<SparseAccessCast<Txn, FE>>,
        R: Into<SparseAccessCast<Txn, FE>>,
    {
        let left = left.into();
        let right = right.into();

        if left.shape() == right.shape() {
            Ok(Self {
                left,
                right,
                block_op,
                value_op,
            })
        } else {
            Err(bad_request!(
                "cannot compare {:?} with {:?} (wrong shape)",
                left,
                right
            ))
        }
    }
}

impl<Txn, FE, T> TensorInstance for SparseCompareLeft<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.left.shape(), self.right.shape());
        self.left.shape()
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseCompareLeft<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
    T: CDatatype + DType,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();

        let source_blocks = self
            .left
            .merge_blocks_inner(self.right, txn_id, range, order)
            .await?;

        let blocks = source_blocks
            .map(move |result| {
                let (coords, (left, right)) = result?;
                let values = (self.block_op)(left, right)?;
                Ok((coords, values))
            })
            .try_filter_map(
                move |(coords, values)| async move { filter_zeros(coords, values, ndim) },
            );

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        let (left, right) = try_join!(
            self.left.read_value(txn_id, coord.to_vec()),
            self.right.read_value(txn_id, coord)
        )?;

        Ok((self.value_op)(left, right))
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseCompareLeft<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these locks in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range.clone()).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, T: CDatatype> From<SparseCompareLeft<Txn, FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(compare: SparseCompareLeft<Txn, FE, T>) -> Self {
        Self::CompareLeft(Box::new(compare))
    }
}

impl<Txn, FE, T> fmt::Debug for SparseCompareLeft<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} and {:?}", self.left, self.right)
    }
}

pub struct SparseCompareConst<Txn, FE, T: CDatatype> {
    left: SparseAccessCast<Txn, FE>,
    right: Number,
    block_op: fn(Block, Number) -> TCResult<Array<T>>,
    value_op: fn(Number, Number) -> T,
}

impl<Txn, FE, T: CDatatype> Clone for SparseCompareConst<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right,
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CDatatype> SparseCompareConst<Txn, FE, T> {
    pub fn new<L>(
        left: L,
        right: Number,
        block_op: fn(Block, Number) -> TCResult<Array<T>>,
        value_op: fn(Number, Number) -> T,
    ) -> Self
    where
        L: Into<SparseAccessCast<Txn, FE>>,
    {
        Self {
            left: left.into(),
            right,
            block_op,
            value_op,
        }
    }
}

impl<Txn, FE, T> TensorInstance for SparseCompareConst<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.left.shape()
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseCompareConst<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();

        let left_blocks = self.left.blocks(txn_id, range, order).await?;
        let blocks = left_blocks
            .map(move |result| {
                result.and_then(|(coords, block)| {
                    (self.block_op)(block, self.right).map(|values| (coords, values))
                })
            })
            .try_filter_map(
                move |(coords, values)| async move { filter_zeros(coords, values, ndim) },
            );

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let left_elements = self.left.elements(txn_id, range, order).await?;

        let elements = left_elements.map_ok(move |(coord, l)| {
            let value = (self.value_op)(l, self.right);
            (coord, value)
        });

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.left
            .read_value(txn_id, coord)
            .map_ok(move |l| (self.value_op)(l, self.right))
            .await
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseCompareConst<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T: CDatatype> From<SparseCompareConst<Txn, FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(compare: SparseCompareConst<Txn, FE, T>) -> Self {
        Self::CompareConst(Box::new(compare))
    }
}

impl<Txn, FE, T> fmt::Debug for SparseCompareConst<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct SparseCond<Cond, Then, OrElse> {
    cond: Cond,
    then: Then,
    or_else: OrElse,
}

impl<Cond, Then, OrElse> SparseCond<Cond, Then, OrElse>
where
    Cond: TensorInstance + fmt::Debug,
    Then: TensorInstance + fmt::Debug,
    OrElse: TensorInstance + fmt::Debug,
{
    pub fn new(cond: Cond, then: Then, or_else: OrElse) -> TCResult<Self> {
        if cond.dtype() == NumberType::Bool
            && cond.shape() == then.shape()
            && cond.shape() == or_else.shape()
            && then.dtype() == or_else.dtype()
        {
            Ok(Self {
                cond,
                then,
                or_else,
            })
        } else {
            Err(bad_request!(
                "cannot select from {then:?} and {or_else:?} based on {cond:?}"
            ))
        }
    }
}

impl<Cond, Then, OrElse> TensorInstance for SparseCond<Cond, Then, OrElse>
where
    Cond: TensorInstance,
    Then: TensorInstance,
    OrElse: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        debug_assert_eq!(self.then.dtype(), self.or_else.dtype());
        self.then.dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.cond.shape(), self.then.shape());
        debug_assert_eq!(self.cond.shape(), self.or_else.shape());
        self.cond.shape()
    }
}

#[async_trait]
impl<Cond, Then, OrElse, T> SparseInstance for SparseCond<Cond, Then, OrElse>
where
    Cond: SparseInstance<DType = u8>,
    Then: SparseInstance<DType = T>,
    OrElse: SparseInstance<DType = T>,
    T: CDatatype + DType + fmt::Debug,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let shape = self.shape().to_vec();
        let ndim = shape.len();

        let strides = strides_for(&shape, ndim);
        let strides = ArrayBase::<Arc<Vec<_>>>::new(vec![strides.len()], Arc::new(strides))?;

        let (cond, then, or_else) = try_join!(
            self.cond.blocks(txn_id, range.clone(), order.to_vec()),
            self.then.blocks(txn_id, range.clone(), order.to_vec()),
            self.or_else.blocks(txn_id, range, order)
        )?;

        let cond = offsets(strides.clone(), cond);
        let then = offsets(strides.clone(), then);
        let or_else = offsets(strides.clone(), or_else);

        let elements = stream::Select::new(cond, then, or_else);
        let offsets = stream::BlockOffsets::new(elements);

        let dims = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(shape))?;
        let blocks = offsets.map(move |result| {
            let (offsets, values) = result?;
            let coords = offsets_to_coords(offsets.into(), strides.clone(), dims.clone())?;
            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        let (cond, then, or_else) = try_join!(
            self.cond.read_value(txn_id, coord.to_vec()),
            self.then.read_value(txn_id, coord.to_vec()),
            self.or_else.read_value(txn_id, coord)
        )?;

        if cond != 0 {
            Ok(then)
        } else {
            Ok(or_else)
        }
    }
}

#[async_trait]
impl<Cond, Then, OrElse> TensorPermitRead for SparseCond<Cond, Then, OrElse>
where
    Cond: TensorPermitRead,
    Then: TensorPermitRead,
    OrElse: TensorPermitRead,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these locks in-order to reduce the risk of a deadlock
        let mut permit = self.cond.read_permit(txn_id, range.clone()).await?;

        let then = self.then.read_permit(txn_id, range.clone()).await?;
        permit.extend(then);

        let or_else = self.or_else.read_permit(txn_id, range).await?;
        permit.extend(or_else);

        Ok(permit)
    }
}

impl<Txn, FE, Cond, Then, OrElse, T> From<SparseCond<Cond, Then, OrElse>>
    for SparseAccess<Txn, FE, T>
where
    Cond: Into<SparseAccess<Txn, FE, u8>>,
    Then: Into<SparseAccess<Txn, FE, T>>,
    OrElse: Into<SparseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    fn from(cond: SparseCond<Cond, Then, OrElse>) -> Self {
        Self::Cond(Box::new(SparseCond {
            cond: cond.cond.into(),
            then: cond.then.into(),
            or_else: cond.or_else.into(),
        }))
    }
}

impl<Cond, Then, OrElse> fmt::Debug for SparseCond<Cond, Then, OrElse>
where
    Cond: fmt::Debug,
    Then: fmt::Debug,
    OrElse: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "select from {:?} and {:?} based on {:?}",
            self.then, self.or_else, self.cond
        )
    }
}

pub struct SparseCow<FE, T, S> {
    source: S,
    filled: SparseFile<FE, T>,
    zeros: SparseFile<FE, T>,
}

impl<FE, T, S: Clone> Clone for SparseCow<FE, T, S> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            filled: self.filled.clone(),
            zeros: self.zeros.clone(),
        }
    }
}

impl<FE, T, S> SparseCow<FE, T, S> {
    pub fn create(source: S, filled: SparseFile<FE, T>, zeros: SparseFile<FE, T>) -> Self
    where
        S: fmt::Debug,
    {
        debug!("create copy-on-write view of {source:?}");

        Self {
            source,
            filled,
            zeros,
        }
    }

    pub fn into_deltas(self) -> (SparseFile<FE, T>, SparseFile<FE, T>) {
        (self.filled, self.zeros)
    }
}

impl<FE, T, S> TensorInstance for SparseCow<FE, T, S>
where
    FE: ThreadSafe,
    T: CDatatype + DType,
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<FE, T, S> SparseInstance for SparseCow<FE, T, S>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    S: SparseInstance<DType = T>,
    Number: CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        log::debug!(
            "SparseCow::blocks in range {range:?} of {:?} with order {order:?}",
            self.shape()
        );

        self.shape().validate_range(&range)?;
        self.shape().validate_axes(&order, true)?;

        let shape = self.source.shape().to_vec();
        let ndim = shape.len();

        let strides = strides_for(&shape, ndim);
        let strides = ArrayBase::<Arc<Vec<_>>>::new(vec![strides.len()], Arc::new(strides))?;

        #[cfg(debug_assertions)]
        let (source_blocks, filled_blocks, zero_blocks) = {
            let source_blocks = self
                .source
                .blocks(txn_id, range.clone(), order.to_vec())
                .await?;

            log::trace!("constructed source block stream");

            let filled_blocks = self
                .filled
                .blocks(txn_id, range.clone(), order.to_vec())
                .await?;

            log::trace!("constructed filled block stream");

            let zero_blocks = self.zeros.blocks(txn_id, range, order).await?;
            log::trace!("constructed zero block stream");

            (source_blocks, filled_blocks, zero_blocks)
        };

        #[cfg(not(debug_assertions))]
        let (source_blocks, filled_blocks, zero_blocks) = try_join!(
            self.source.blocks(txn_id, range.clone(), order.to_vec()),
            self.filled.blocks(txn_id, range.clone(), order.to_vec()),
            self.zeros.blocks(txn_id, range, order)
        )?;

        let source_elements = offsets(strides.clone(), source_blocks);
        let filled_elements = offsets(strides.clone(), filled_blocks);
        let zero_elements = offsets(strides.clone(), zero_blocks);

        let elements = stream::TryDiff::new(source_elements, zero_elements);
        let elements = stream::TryMerge::new(elements, filled_elements);
        let offsets = stream::BlockOffsets::new(elements);

        let dims = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(shape))?;
        let blocks = offsets.map(move |result| {
            let (offsets, values) = result?;
            log::trace!("block has {} values", values.size());

            let coords = offsets_to_coords(offsets.into(), strides.clone(), dims.clone())?;

            Ok((coords, values))
        });

        log::trace!("constructed SparseCow block stream");

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;

        let key = coord.iter().copied().map(Number::from).collect();

        {
            let zeros = self.zeros.table().read().await;
            if zeros.contains(&key).await? {
                return Ok(Self::DType::zero());
            }
        }

        {
            let filled = self.filled.table().read().await;
            if let Some(mut row) = filled.get_row(key).await? {
                let value = row.pop().expect("value");
                return Ok(value.cast_into());
            }
        }

        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<FE, T, S> TensorPermitRead for SparseCow<FE, T, S>
where
    FE: Send + Sync,
    T: CDatatype + DType,
    S: TensorPermitRead,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.source.read_permit(txn_id, range).await
    }
}

#[async_trait]
impl<FE, T, S> TensorPermitWrite for SparseCow<FE, T, S>
where
    FE: Send + Sync,
    T: CDatatype + DType,
    S: TensorPermitWrite,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.source.write_permit(txn_id, range).await
    }
}

#[async_trait]
impl<'a, FE, T, S> SparseWriteLock<'a> for SparseCow<FE, T, S>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    S: SparseInstance<DType = T> + Clone,
    Number: From<T> + CastInto<T>,
{
    type Guard = SparseCowWriteGuard<'a, FE, T, S>;

    async fn write(&'a self) -> Self::Guard {
        debug!("lock {self:?} for writing...");

        SparseCowWriteGuard {
            source: &self.source,
            filled: self.filled.write().await,
            zeros: self.zeros.write().await,
        }
    }
}

impl<Txn, FE, T, S> From<SparseCow<FE, T, S>> for SparseAccess<Txn, FE, T>
where
    T: CDatatype,
    S: Into<SparseAccess<Txn, FE, T>>,
{
    fn from(cow: SparseCow<FE, T, S>) -> Self {
        SparseAccess::Cow(Box::new(SparseCow {
            source: cow.source.into(),
            filled: cow.filled,
            zeros: cow.zeros,
        }))
    }
}

impl<FE, T, S> fmt::Debug for SparseCow<FE, T, S>
where
    FE: ThreadSafe,
    T: CDatatype + DType,
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "copy-on-write view of {:?}", self.source)
    }
}

pub struct SparseCowWriteGuard<'a, FE, T, S> {
    source: &'a S,
    filled: SparseFileWriteGuard<'a, FE, T>,
    zeros: SparseFileWriteGuard<'a, FE, T>,
}

#[async_trait]
impl<'a, FE, T, S> SparseWriteGuard<T> for SparseCowWriteGuard<'a, FE, T, S>
where
    FE: AsType<Node> + ThreadSafe,
    S: SparseInstance<DType = T> + Clone,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T>,
{
    async fn clear(&mut self, txn_id: TxnId, range: Range) -> TCResult<()> {
        debug!("clear {range:?} of COW guard with source {:?}", self.source);

        self.filled.clear(txn_id, range.clone()).await?;
        trace!("cleared filled elements");

        self.zeros.clear(txn_id, range.clone()).await?;
        trace!("cleared zero elements");

        trace!("copying new zero elements from {:?}...", self.source);

        let mut elements = self
            .source
            .clone()
            .elements(txn_id, range, Axes::default())
            .await?;

        while let Some((coord, value)) = elements.try_next().await? {
            self.zeros.write_value(txn_id, coord, value).await?;
        }

        trace!("copied new zero elements");

        Ok(())
    }

    async fn overwrite<O: SparseInstance<DType = T>>(
        &mut self,
        txn_id: TxnId,
        other: O,
    ) -> TCResult<()> {
        if self.source.shape() != other.shape() {
            return Err(bad_request!(
                "cannot overwrite a sparse tensor of shape {:?} with one of shape {:?}",
                self.source.shape(),
                other.shape()
            ));
        }

        self.clear(txn_id, Range::default()).await?;

        let mut elements = other
            .elements(txn_id, Range::default(), Axes::default())
            .await?;

        while let Some((coord, value)) = elements.try_next().await? {
            self.write_value(txn_id, coord, value).await?;
        }

        Ok(())
    }

    async fn write_value(&mut self, txn_id: TxnId, coord: Coord, value: T) -> Result<(), TCError> {
        let inverse = if value == T::zero() {
            T::one()
        } else {
            T::zero()
        };

        try_join!(
            self.filled.write_value(txn_id, coord.to_vec(), value),
            self.zeros.write_value(txn_id, coord, inverse)
        )
        .map(|_| ())
    }
}

pub struct SparseDense<Txn, FE, T: CDatatype> {
    source: DenseAccess<Txn, FE, T>,
}

impl<Txn, FE, T: CDatatype> Clone for SparseDense<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
        }
    }
}

impl<Txn, FE, T: CDatatype> SparseDense<Txn, FE, T> {
    pub fn new<S>(source: S) -> Self
    where
        S: Into<DenseAccess<Txn, FE, T>>,
        T: CDatatype,
    {
        Self {
            source: source.into(),
        }
    }
}

impl<Txn, FE, T> TensorInstance for SparseDense<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseDense<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        self.shape().validate_range(&range)?;
        self.shape().validate_axes(&order, true)?;

        let is_slice = !self.shape().is_covered_by(&range);
        let range = range.normalize(self.shape());

        let source = if order.iter().copied().enumerate().all(|(i, x)| i == x) {
            Ok(self.source)
        } else {
            DenseTranspose::new(self.source, Some(order)).map(DenseAccess::from)
        }?;

        let ndim = source.ndim();
        let coord_block_size = source.block_size() * ndim;

        let coords = range.affected().map(|coord| coord.into_iter()).flatten();
        let coords = futures::stream::iter(coords).chunks(coord_block_size);

        let source_blocks: Blocks<_, Array<Self::DType>> = if is_slice {
            let source_blocks = DenseSlice::new(source, range)?.read_blocks(txn_id).await?;

            let blocks = coords
                .zip(source_blocks)
                .map(|(coords, values)| values.map(|values| (coords, values.into())));

            Box::pin(blocks)
        } else {
            let source_blocks = source.read_blocks(txn_id).await?;
            let blocks = coords
                .zip(source_blocks)
                .map(|(coords, values)| values.map(|values| (coords, values.into())));

            Box::pin(blocks)
        };

        let zero = Self::DType::zero();
        let blocks = source_blocks.try_filter_map(move |(coords, values)| async move {
            let queue = autoqueue(&values)?;
            let values = values.read(&queue)?.to_slice()?;

            let (coords, values) = coords
                .into_par_iter()
                .chunks(ndim)
                .zip(values.as_ref().into_par_iter().copied())
                .filter(|(_coord, value)| value != &zero)
                .map(|(coord, value)| (coord, value))
                .unzip::<_, _, Vec<Vec<u64>>, Vec<Self::DType>>();

            if values.is_empty() {
                Ok(None)
            } else {
                let num_values = values.len();
                let coords = coords.into_iter().flatten().collect();
                let coords = ArrayBase::<Vec<u64>>::new(vec![num_values, ndim], coords)?;
                let values = ArrayBase::<Vec<Self::DType>>::new(vec![num_values], values)?;
                Ok(Some((coords, values)))
            }
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseDense<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T: CDatatype> From<SparseDense<Txn, FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(dense: SparseDense<Txn, FE, T>) -> Self {
        Self::Dense(Box::new(dense))
    }
}

impl<Txn, FE, T: CDatatype> fmt::Debug for SparseDense<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "sparse view of {:?}", self.source)
    }
}

#[derive(Clone)]
pub struct SparseExpand<S> {
    source: S,
    transform: Expand,
}

impl<S: TensorInstance + fmt::Debug> SparseExpand<S> {
    pub fn new(source: S, axes: Axes) -> TCResult<Self> {
        debug!("expand axes {axes:?} of {source:?}");
        Expand::new(source.shape().clone(), axes).map(|transform| Self { source, transform })
    }
}

impl<S: TensorInstance> TensorInstance for SparseExpand<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseExpand<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(txn_id, range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        debug!("SparseExpand::elements in range {range:?} with order {order:?}");

        self.shape().validate_range(&range)?;
        self.shape().validate_axes(&order, true)?;

        let source_range = self.transform.invert_range(range);
        let source_order = self.transform.invert_axes(order);

        trace!("source range is {source_range:?} and order is {source_order:?}");

        let source_elements = self
            .source
            .elements(txn_id, source_range, source_order)
            .await?;

        let elements = source_elements.map_ok(move |(source_coord, value)| {
            let coord = self.transform.map_coord(source_coord);
            (coord, value)
        });

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, source_coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for SparseExpand<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T, S> From<SparseExpand<S>> for SparseAccess<Txn, FE, T>
where
    T: CDatatype,
    S: Into<SparseAccess<Txn, FE, T>>,
{
    fn from(expand: SparseExpand<S>) -> Self {
        Self::Expand(Box::new(SparseExpand {
            source: expand.source.into(),
            transform: expand.transform,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseExpand<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "expand axes {:?} of {:?}",
            self.transform.expand_axes(),
            self.source,
        )
    }
}

pub struct SparseReduce<Txn, FE, T: CDatatype> {
    reductor: fn(TxnId, SparseSlice<SparseAccess<Txn, FE, T>>) -> TCBoxTryFuture<'static, T>,
    source: SparseAccess<Txn, FE, T>,
    transform: Arc<Reduce>,
}

impl<Txn, FE, T: CDatatype> Clone for SparseReduce<Txn, FE, T> {
    fn clone(&self) -> Self {
        SparseReduce {
            reductor: self.reductor,
            source: self.source.clone(),
            transform: self.transform.clone(),
        }
    }
}

impl<Txn, FE, T> SparseReduce<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    pub fn new<S>(
        source: S,
        mut axes: Axes,
        keepdims: bool,
        reductor: fn(TxnId, SparseSlice<SparseAccess<Txn, FE, T>>) -> TCBoxTryFuture<'static, T>,
    ) -> TCResult<Self>
    where
        SparseAccess<Txn, FE, T>: From<S>,
    {
        axes.sort();
        axes.dedup();

        let source = SparseAccess::from(source);

        log::debug!("SparseReduce::new axes {axes:?} of {source:?}");

        Reduce::new(source.shape().clone(), axes, keepdims)
            .map(Arc::new)
            .map(|transform| Self {
                reductor,
                source,
                transform,
            })
    }
}

impl<Txn, FE, T> SparseReduce<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CDatatype + DType,
{
    async fn reduce_element(&self, txn_id: TxnId, coord: Coord) -> TCResult<(Coord, T)>
    where
        T: fmt::Debug,
    {
        self.shape().validate_coord(&coord)?;

        let source_range = self.transform.invert_coord(&coord);
        let slice = SparseSlice::new(self.source.clone(), source_range.into())?;
        (self.reductor)(txn_id, slice)
            .map_ok(|reduced| (coord, reduced))
            .await
    }
}

impl<Txn, FE, T> TensorInstance for SparseReduce<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseReduce<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(txn_id, range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        mut order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        order.reserve(self.ndim() - order.len());

        for x in 0..self.ndim() {
            if !order.contains(&x) {
                order.push(x);
            }
        }

        self.transform.shape().validate_range(&range)?;
        self.transform.shape().validate_axes(&order, true)?;

        let source_range = self.transform.invert_range(range);
        let source_order = self.transform.invert_axes(order);
        let source_axes = (0..self.source.ndim())
            .into_iter()
            .filter(|x| !self.transform.axes().contains(x))
            .collect();

        let filled_at = self
            .source
            .clone()
            .filled_at(txn_id, source_range, source_order, source_axes)
            .await?;

        let zero = T::zero();
        let elements = filled_at
            .map(move |result| {
                let coord = result?;
                let this = self.clone();
                Ok(async move { this.reduce_element(txn_id, coord).await })
            })
            .try_buffered(num_cpus::get())
            .try_filter(move |(_coord, value)| futures::future::ready(*value != zero));

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.reduce_element(txn_id, coord)
            .map_ok(|(_coord, value)| value)
            .await
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseReduce<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T> From<SparseReduce<Txn, FE, T>> for SparseAccess<Txn, FE, T>
where
    T: CDatatype,
{
    fn from(reduce: SparseReduce<Txn, FE, T>) -> Self {
        Self::Reduce(Box::new(reduce))
    }
}

impl<Txn, FE, T> fmt::Debug for SparseReduce<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "reduce axes {:?} of {:?}",
            self.transform.reduce_axes(),
            self.source
        )
    }
}

#[derive(Clone)]
pub struct SparseReshape<S> {
    source: S,
    transform: Reshape,
}

impl<S: SparseInstance> SparseReshape<S> {
    pub fn new(source: S, shape: Shape) -> TCResult<Self> {
        Reshape::new(source.shape().clone(), shape).map(|transform| Self { source, transform })
    }
}

impl<S: TensorInstance> TensorInstance for SparseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseReshape<S> {
    type CoordBlock = Array<u64>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = S::DType;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        self.shape().validate_range(&range)?;
        self.shape().validate_axes(&order, true)?;

        let source_range = if range.is_empty() {
            Ok(range)
        } else {
            Err(bad_request!(
                "cannot slice a reshaped sparse tensor (consider making a copy first)"
            ))
        }?;

        let source_order = if order
            .iter()
            .copied()
            .zip(0..self.ndim())
            .all(|(x, o)| x == o)
        {
            Ok(order)
        } else {
            Err(bad_request!(
                "cannot transpose a reshaped sparse tensor (consider making a copy first)"
            ))
        }?;

        let source_ndim = self.source.ndim();
        let source_blocks = self
            .source
            .blocks(txn_id, source_range, source_order)
            .await?;

        let source_strides = Arc::new(self.transform.source_strides().to_vec());
        let source_strides = ArrayBase::<Arc<Vec<_>>>::new(vec![source_ndim], source_strides)?;

        let ndim = self.transform.shape().len();
        let strides = Arc::new(self.transform.strides().to_vec());
        let strides = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], strides)?;
        let dims = Arc::new(self.transform.shape().to_vec());
        let dims = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], dims)?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;

            debug_assert_eq!(source_coords.size() % source_ndim, 0);
            debug_assert_eq!(source_coords.size() / source_ndim, values.size());

            let source_strides = source_strides
                .clone()
                .broadcast(vec![values.size(), source_ndim])?;

            let offsets = source_coords.mul(source_strides)?;
            let offsets = offsets.sum(vec![1], false)?;

            let coords = offsets_to_coords(offsets.into(), strides.clone(), dims.clone())?;

            Result::<_, TCError>::Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, source_coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for SparseReshape<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        if self.transform.shape().is_covered_by(&range) {
            self.source.read_permit(txn_id, Range::default()).await
        } else {
            Err(bad_request!(
                "cannot lock range {:?} of {:?} for reading (consider making a copy first)",
                range,
                self
            ))
        }
    }
}

impl<Txn, FE, T, S> From<SparseReshape<S>> for SparseAccess<Txn, FE, T>
where
    T: CDatatype,
    S: Into<SparseAccess<Txn, FE, T>>,
{
    fn from(reshape: SparseReshape<S>) -> Self {
        Self::Reshape(Box::new(SparseReshape {
            source: reshape.source.into(),
            transform: reshape.transform,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseReshape<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "reshape {:?} into {:?}",
            self.source,
            self.transform.shape()
        )
    }
}

#[derive(Clone)]
pub struct SparseSlice<S> {
    source: S,
    transform: Slice,
}

impl<S> SparseSlice<S>
where
    S: TensorInstance + fmt::Debug,
{
    pub fn new(source: S, range: Range) -> TCResult<Self> {
        debug!("SparseSlice::new range {range:?} of {source:?}");
        let transform = Slice::new(source.shape().clone(), range)?;
        trace!("SparseSlice shape is {:?}", transform.shape());
        Ok(Self { source, transform })
    }

    fn source_order(&self, order: Axes) -> Result<Axes, TCError> {
        self.shape().validate_axes(&order, true)?;

        let mut source_axes = Vec::with_capacity(self.ndim());
        for (x, bound) in self.transform.range().iter().enumerate() {
            if !bound.is_index() {
                source_axes.push(x);
            }
        }

        Ok(order.into_iter().map(|x| source_axes[x]).collect())
    }
}

impl<S> TensorInstance for SparseSlice<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S> SparseInstance for SparseSlice<S>
where
    S: SparseInstance,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = S::DType;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        self.shape().validate_range(&range)?;
        let source_order = self.source_order(order)?;

        let source_range = self.transform.invert_range(range);
        let source_blocks = self
            .source
            .blocks(txn_id, source_range, source_order)
            .await?;

        let transform = self.transform;
        let ndim = transform.shape().len();
        let source_ndim = transform.source_shape().len();

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;

            debug_assert_eq!(source_coords.shape(), [values.size(), source_ndim]);

            let queue = autoqueue(&source_coords)?;
            let source_coords = source_coords.read(&queue)?;

            let coords = source_coords
                .to_slice()?
                .as_ref()
                .par_iter()
                .copied()
                .chunks(source_ndim)
                .map(|source_coord| transform.map_coord(source_coord))
                .flatten()
                .collect();

            let coords = ArrayBase::<Vec<u64>>::new(vec![values.size(), ndim], coords)?;

            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        debug!("SparseSlice::elements in {range:?} with order {order:?}");

        self.shape().validate_range(&range)?;

        let source_order = self.source_order(order)?;

        let source_range = self.transform.invert_range(range);
        trace!(
            "the range within the source of {:?} is {source_range:?}",
            self
        );

        let source_elements = self
            .source
            .elements(txn_id, source_range, source_order)
            .await?;

        let elements =
            source_elements.map_ok(move |(coord, value)| (self.transform.map_coord(coord), value));

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, source_coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for SparseSlice<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

#[async_trait]
impl<S: TensorPermitWrite> TensorPermitWrite for SparseSlice<S> {
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.write_permit(txn_id, range).await
    }
}

#[async_trait]
impl<'a, S> SparseWriteLock<'a> for SparseSlice<S>
where
    S: SparseWriteLock<'a>,
{
    type Guard = SparseSliceWriteGuard<'a, S::Guard, S::DType>;

    async fn write(&'a self) -> SparseSliceWriteGuard<'a, S::Guard, S::DType> {
        SparseSliceWriteGuard {
            transform: &self.transform,
            guard: self.source.write().await,
            dtype: PhantomData,
        }
    }
}

impl<Txn, FE, T, S> From<SparseSlice<S>> for SparseAccess<Txn, FE, T>
where
    T: CDatatype,
    S: Into<SparseAccess<Txn, FE, T>>,
{
    fn from(slice: SparseSlice<S>) -> Self {
        Self::Slice(Box::new(SparseSlice {
            source: slice.source.into(),
            transform: slice.transform,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "slice {:?} from {:?}",
            self.transform.range(),
            self.source,
        )
    }
}

pub struct SparseSliceWriteGuard<'a, G, T> {
    transform: &'a Slice,
    guard: G,
    dtype: PhantomData<T>,
}

#[async_trait]
impl<'a, G, T> SparseWriteGuard<T> for SparseSliceWriteGuard<'a, G, T>
where
    G: SparseWriteGuard<T>,
    T: CDatatype + DType,
{
    async fn clear(&mut self, txn_id: TxnId, range: Range) -> TCResult<()> {
        self.transform.shape().validate_range(&range)?;
        self.guard
            .clear(txn_id, self.transform.invert_range(range))
            .await
    }

    async fn overwrite<O: SparseInstance<DType = T>>(
        &mut self,
        txn_id: TxnId,
        other: O,
    ) -> TCResult<()> {
        if self.transform.shape() != other.shape() {
            return Err(bad_request!(
                "cannot overwrite a sparse tensor of shape {:?} with one of shape {:?}",
                self.transform.shape(),
                other.shape()
            ));
        }

        self.clear(txn_id, Range::default()).await?;

        let mut elements = other
            .elements(txn_id, Range::default(), Axes::default())
            .await?;

        while let Some((coord, value)) = elements.try_next().await? {
            self.write_value(txn_id, coord, value).await?;
        }

        Ok(())
    }

    async fn write_value(&mut self, txn_id: TxnId, coord: Coord, value: T) -> Result<(), TCError> {
        self.transform.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord);
        self.guard.write_value(txn_id, coord, value).await
    }
}

#[derive(Clone)]
pub struct SparseTranspose<S> {
    source: S,
    transform: Transpose,
}

impl<S: SparseInstance + fmt::Debug> SparseTranspose<S> {
    pub fn new(source: S, permutation: Option<Axes>) -> TCResult<Self> {
        log::debug!("SparseTranspose::new({source:?}, {permutation:?})");

        Transpose::new(source.shape().clone(), permutation)
            .map(|transform| Self { source, transform })
    }
}

impl<S> TensorInstance for SparseTranspose<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S> SparseInstance for SparseTranspose<S>
where
    S: SparseInstance,
    <S::CoordBlock as NDArrayTransform>::Transpose: NDArrayRead<DType = u64> + Into<Array<u64>>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = S::DType;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        debug!("SparseTranspose::blocks in {range:?} with order {order:?}");

        self.shape().validate_range(&range)?;
        self.shape().validate_axes(&order, true)?;

        let range = range.normalize(self.shape());
        debug_assert_eq!(range.len(), self.ndim());

        let order = if order.is_empty() {
            (0..self.ndim()).into_iter().collect()
        } else {
            order
        };

        let source_range = self.transform.invert_range(&range);
        let source_order = self.transform.invert_axes(order);

        trace!("SparseTranspose source range is {source_range:?} and order is {source_order:?}");

        let source_blocks = self
            .source
            .blocks(txn_id, source_range, source_order)
            .await?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;

            let queue = autoqueue(&source_coords)?;
            let source_coords = source_coords.read(&queue)?.to_slice()?;

            let ndim = self.transform.shape().len();
            let coords = source_coords
                .into_vec()
                .into_par_iter()
                .chunks(ndim)
                .map(|source_coord| self.transform.map_coord(source_coord))
                .flatten()
                .collect();

            let coords = ArrayBase::<Vec<u64>>::new(vec![values.size(), ndim], coords)?;

            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, source_coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for SparseTranspose<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(&range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T, S> From<SparseTranspose<S>> for SparseAccess<Txn, FE, T>
where
    T: CDatatype,
    S: Into<SparseAccess<Txn, FE, T>>,
{
    fn from(transpose: SparseTranspose<S>) -> Self {
        Self::Transpose(Box::new(SparseTranspose {
            source: transpose.source.into(),
            transform: transpose.transform,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseTranspose<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "transpose of {:?} with permutation {:?}",
            self.source,
            self.transform.axes()
        )
    }
}

#[derive(Clone)]
pub struct SparseUnary<S, T: CDatatype> {
    source: S,
    block_op: fn(Array<T>) -> TCResult<Array<T>>,
    value_op: fn(T) -> T,
}

impl<S, T: CDatatype> SparseUnary<S, T> {
    pub fn new(
        source: S,
        block_op: fn(Array<T>) -> TCResult<Array<T>>,
        value_op: fn(T) -> T,
    ) -> Self {
        Self {
            source,
            block_op,
            value_op,
        }
    }
}

impl<S: TensorInstance, T: CDatatype + DType> TensorInstance for SparseUnary<S, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<S: SparseInstance<DType = T>, T: CDatatype + DType> SparseInstance for SparseUnary<S, T> {
    type CoordBlock = S::CoordBlock;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        debug!("SparseUnary::blocks in {range:?} with order {order:?}");

        let source_blocks = self.source.blocks(txn_id, range, order).await?;
        let blocks = source_blocks.map(move |result| {
            let (coords, values) = result?;
            (self.block_op)(values.into()).map(|values| (coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.source
            .read_value(txn_id, coord)
            .map_ok(|value| (self.value_op)(value))
            .await
    }
}

#[async_trait]
impl<S: TensorPermitRead, T: CDatatype> TensorPermitRead for SparseUnary<S, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<SparseUnary<S, T>> for SparseAccess<Txn, FE, T>
where
    S: Into<SparseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    fn from(unary: SparseUnary<S, T>) -> Self {
        Self::Unary(Box::new(SparseUnary {
            source: unary.source.into(),
            block_op: unary.block_op,
            value_op: unary.value_op,
        }))
    }
}

impl<S: fmt::Debug, T: CDatatype + DType> fmt::Debug for SparseUnary<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary operation on {:?}", self.source)
    }
}

pub struct SparseUnaryCast<Txn, FE, T: CDatatype> {
    source: SparseAccessCast<Txn, FE>,
    block_op: fn(Block) -> TCResult<Array<T>>,
    value_op: fn(Number) -> T,
}

impl<Txn, FE, T: CDatatype> Clone for SparseUnaryCast<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CDatatype> SparseUnaryCast<Txn, FE, T> {
    pub fn new<S: Into<SparseAccessCast<Txn, FE>>>(
        source: S,
        block_op: fn(Block) -> TCResult<Array<T>>,
        value_op: fn(Number) -> T,
    ) -> Self {
        Self {
            source: source.into(),
            block_op,
            value_op,
        }
    }
}

macro_rules! block_f32_cast {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Block::F32($var) => $call,
            Block::I16($var) => $call,
            Block::I32($var) => $call,
            Block::U8($var) => $call,
            Block::U16($var) => $call,
            Block::U32($var) => $call,
            block => unreachable!("32-bit float op on {:?}", block),
        }
    };
}

impl<Txn, FE> SparseUnaryCast<Txn, FE, f32> {
    pub fn asin_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.asin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).asin(),
        }
    }

    pub fn sin_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.sin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).sin(),
        }
    }

    pub fn sinh_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.sinh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).sinh(),
        }
    }

    pub fn acos_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.acos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).acos(),
        }
    }

    pub fn cos_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.cos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).cos(),
        }
    }

    pub fn cosh_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.cosh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).cosh(),
        }
    }

    pub fn atan_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.atan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).atan(),
        }
    }

    pub fn tan_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.tan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).tan(),
        }
    }

    pub fn tanh_f32<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.tanh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f32::cast_from(n).tanh(),
        }
    }
}

macro_rules! block_f64_cast {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Block::F64($var) => $call,
            Block::I64($var) => $call,
            Block::U64($var) => $call,
            block => unreachable!("64-bit float op on {:?}", block),
        }
    };
}

impl<Txn, FE> SparseUnaryCast<Txn, FE, f64> {
    pub fn asin_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.asin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).asin(),
        }
    }

    pub fn sin_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.sin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).sin(),
        }
    }

    pub fn sinh_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.sinh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).sinh(),
        }
    }

    pub fn acos_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.acos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).acos(),
        }
    }

    pub fn cos_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.cos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).cos(),
        }
    }

    pub fn cosh_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.cosh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).cosh(),
        }
    }

    pub fn atan_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.atan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).atan(),
        }
    }

    pub fn tan_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.tan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).tan(),
        }
    }

    pub fn tanh_f64<S: Into<SparseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.tanh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| f64::cast_from(n).tanh(),
        }
    }
}

impl<Txn, FE, T> TensorInstance for SparseUnaryCast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let source = &self.source;
        access_cast_dispatch!(source, this, this.shape())
    }
}

#[async_trait]
impl<Txn, FE, T> SparseInstance for SparseUnaryCast<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
    T: CDatatype + DType,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        debug!("SparseUnaryCast::blocks in range {range:?} with order {order:?}");

        let source_blocks = self.source.blocks(txn_id, range, order).await?;
        let blocks = source_blocks.map(move |result| {
            let (coords, values) = result?;

            (self.block_op)(values)
                .map(|values| (coords, values.into()))
                .map_err(TCError::from)
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(txn_id, range, order).await?;
        block_elements(blocks, ndim)
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.source
            .read_value(txn_id, coord)
            .map_ok(|value| (self.value_op)(value))
            .await
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for SparseUnaryCast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        let source = &self.source;
        access_cast_dispatch!(source, this, this.read_permit(txn_id, range).await)
    }
}

impl<Txn, FE, T: CDatatype> From<SparseUnaryCast<Txn, FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(unary: SparseUnaryCast<Txn, FE, T>) -> Self {
        Self::UnaryCast(Box::new(unary))
    }
}

impl<Txn, FE, T> fmt::Debug for SparseUnaryCast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary operation on {:?}", self.source)
    }
}

pub struct SparseVersion<FE, T> {
    file: SparseFile<FE, T>,
    semaphore: Semaphore,
}

impl<FE, T> Clone for SparseVersion<FE, T> {
    fn clone(&self) -> Self {
        Self {
            file: self.file.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<FE, T> SparseVersion<FE, T> {
    pub fn new(file: SparseFile<FE, T>, semaphore: Semaphore) -> Self {
        Self { file, semaphore }
    }

    pub fn commit(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, false);
    }

    pub fn rollback(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, false);
    }

    pub fn finalize(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, true);
    }
}

impl<FE, T> TensorInstance for SparseVersion<FE, T>
where
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        self.file.dtype()
    }

    fn shape(&self) -> &Shape {
        self.file.shape()
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseVersion<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = <SparseFile<FE, T> as SparseInstance>::CoordBlock;
    type ValueBlock = <SparseFile<FE, T> as SparseInstance>::ValueBlock;
    type Blocks = <SparseFile<FE, T> as SparseInstance>::Blocks;
    type DType = <SparseFile<FE, T> as SparseInstance>::DType;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        self.file.blocks(txn_id, range, order).await
    }

    async fn elements(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        self.file.elements(txn_id, range, order).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.file.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<FE, T> TensorPermitRead for SparseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.semaphore
            .read(txn_id, range)
            .map_ok(|permit| vec![permit])
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<FE, T> TensorPermitWrite for SparseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.semaphore
            .write(txn_id, range)
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<'a, FE, T> SparseWriteLock<'a> for SparseVersion<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Guard = <SparseFile<FE, T> as SparseWriteLock<'a>>::Guard;

    async fn write(&'a self) -> Self::Guard {
        self.file.write().await
    }
}

impl<Txn, FE, T: CDatatype> From<SparseVersion<FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(version: SparseVersion<FE, T>) -> Self {
        Self::Version(version)
    }
}

impl<FE, T> fmt::Debug for SparseVersion<FE, T>
where
    FE: ThreadSafe,
    T: CDatatype + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional version of {:?}", self.file)
    }
}

#[inline]
fn block_elements<T: CDatatype, C: NDArrayRead<DType = u64>, V: NDArrayRead<DType = T>>(
    blocks: impl Stream<Item = TCResult<(C, V)>> + Send + 'static,
    ndim: usize,
) -> TCResult<Elements<T>> {
    let elements = blocks
        .map(move |result| {
            let (coords, values) = result?;

            let queue = autoqueue(&coords)?;
            let coords = coords.read(&queue)?.to_slice()?;

            let queue = autoqueue(&values)?;
            let values = values.read(&queue)?.to_slice()?;

            let tuples = coords
                .as_ref()
                .into_par_iter()
                .copied()
                .chunks(ndim)
                .zip(values.as_ref().into_par_iter().copied())
                .map(Ok)
                .collect::<Vec<_>>();

            Result::<_, TCError>::Ok(futures::stream::iter(tuples))
        })
        .try_flatten();

    Ok(Box::pin(elements))
}

#[inline]
fn offsets<C, V, T>(
    strides: ArrayBase<Arc<Vec<u64>>>,
    blocks: impl Stream<Item = Result<(C, V), TCError>> + Send + 'static,
) -> impl Stream<Item = Result<(u64, T), TCError>> + Send
where
    C: NDArrayRead<DType = u64> + NDArrayMath + 'static,
    V: NDArrayRead<DType = T>,
    T: CDatatype,
{
    let offsets = blocks
        .map(move |result| {
            let (coords, values) = result?;

            let queue = autoqueue(&coords)?;
            let strides = strides.clone().broadcast(coords.shape().to_vec())?;
            let offsets = coords.mul(strides)?.sum(vec![1], false)?;
            let offsets = offsets.read(&queue)?.to_slice()?.into_vec();

            let queue = autoqueue(&values)?;
            let values = values.read(&queue)?.to_slice()?.into_vec();

            debug_assert_eq!(offsets.len(), values.len());

            Result::<_, TCError>::Ok(futures::stream::iter(
                offsets
                    .into_iter()
                    .zip(values)
                    .map(Result::<_, TCError>::Ok),
            ))
        })
        .try_flatten();

    Box::pin(offsets)
}

#[inline]
fn filter_zeros<T: CDatatype>(
    coords: Array<u64>,
    values: Array<T>,
    ndim: usize,
) -> TCResult<Option<(Array<u64>, Array<T>)>> {
    let zero = T::zero();

    let values = ArrayBase::<Vec<T>>::copy(&values)?;

    if values.all()? {
        Ok(Some((coords, values.into())))
    } else {
        let queue = autoqueue(&coords)?;
        let coord_slice = coords.read(&queue)?.to_slice()?;
        let value_slice = values.into_inner();
        debug_assert_eq!(coord_slice.len() % ndim, 0);

        let (filtered_coords, filtered_values): (Vec<&[u64]>, Vec<T>) = coord_slice
            .as_ref()
            .par_chunks(ndim)
            .zip(value_slice.into_par_iter())
            .filter_map(|(coord, value)| {
                if value == zero {
                    None
                } else {
                    Some((coord, value))
                }
            })
            .unzip();

        let filtered_coords = filtered_coords
            .into_par_iter()
            .map(|coord| coord.into_par_iter().copied())
            .flatten()
            .collect();

        if filtered_values.is_empty() {
            Ok(None)
        } else {
            let num_values = filtered_values.len();
            let coords = ArrayBase::<Vec<u64>>::new(vec![ndim, num_values], filtered_coords)?;
            let values = ArrayBase::<Vec<T>>::new(vec![num_values], filtered_values)?;

            Ok(Some((coords.into(), values.into())))
        }
    }
}

async fn merge_blocks_inner<L, R, T>(
    left: L,
    right: R,
    txn_id: TxnId,
    shape: Shape,
    range: Range,
    order: Axes,
) -> TCResult<
    impl Stream<Item = TCResult<(Array<u64>, (ArrayBase<Vec<T>>, ArrayBase<Vec<T>>))>> + Send,
>
where
    L: SparseInstance<DType = T>,
    R: SparseInstance<DType = T>,
    T: CDatatype + fmt::Debug,
{
    debug_assert_eq!(&shape, left.shape());
    debug_assert_eq!(&shape, right.shape());

    let strides = strides_for(&shape, shape.len());
    let strides = ArrayBase::<Arc<Vec<u64>>>::new(vec![strides.len()], Arc::new(strides))?;
    let dims = ArrayBase::<Arc<Vec<u64>>>::new(vec![shape.len()], Arc::new(shape.to_vec()))?;

    let (left_blocks, right_blocks) = try_join!(
        left.blocks(txn_id, range.clone(), order.to_vec()),
        right.blocks(txn_id, range, order)
    )?;

    let left = offsets(strides.clone(), left_blocks);
    let right = offsets(strides.clone(), right_blocks);

    let elements = stream::InnerJoin::new(left, right);
    let blocks = stream::BlockOffsetsDual::new(elements);
    let coord_blocks = blocks.map(move |result| {
        let (offsets, values) = result?;
        let coords = offsets_to_coords(offsets.into(), strides.clone(), dims.clone())?;
        Ok((coords, values))
    });

    Ok(coord_blocks)
}

pub(super) async fn merge_blocks_outer<L, R, T>(
    left: L,
    right: R,
    txn_id: TxnId,
    shape: Shape,
    range: Range,
    order: Axes,
) -> TCResult<
    impl Stream<Item = TCResult<(Array<u64>, (ArrayBase<Vec<T>>, ArrayBase<Vec<T>>))>> + Send,
>
where
    L: SparseInstance<DType = T>,
    R: SparseInstance<DType = T>,
    T: CDatatype + fmt::Debug,
{
    debug_assert_eq!(&shape, left.shape());
    debug_assert_eq!(&shape, right.shape());

    let ndim = shape.len();
    let strides = strides_for(&shape, ndim);
    let strides = ArrayBase::<Arc<Vec<u64>>>::new(vec![ndim], Arc::new(strides))?;
    let dims = ArrayBase::<Arc<Vec<u64>>>::new(vec![ndim], Arc::new(shape.to_vec()))?;

    let (left_blocks, right_blocks) = try_join!(
        left.blocks(txn_id, range.clone(), order.to_vec()),
        right.blocks(txn_id, range, order)
    )?;

    let left = offsets(strides.clone(), left_blocks);
    let right = offsets(strides.clone(), right_blocks);

    let elements = stream::OuterJoin::new(left, right, T::zero());
    let blocks = stream::BlockOffsetsDual::new(elements);
    let coord_blocks = blocks.map(move |result| {
        let (offsets, values) = result?;
        let coords = offsets_to_coords(offsets.into(), strides.clone(), dims.clone())?;
        Ok((coords, values))
    });

    Ok(coord_blocks)
}

#[inline]
fn offsets_to_coords(
    offsets: Array<u64>,
    strides: ArrayBase<Arc<Vec<u64>>>,
    dims: ArrayBase<Arc<Vec<u64>>>,
) -> TCResult<Array<u64>> {
    let ndim = dims.size();
    let num_offsets = offsets.size();
    let block_shape = vec![num_offsets, ndim];

    offsets
        .expand_dims(vec![1])?
        .broadcast(block_shape.to_vec())?
        .checked_div(strides.broadcast(block_shape.to_vec())?)?
        .rem(dims.broadcast(block_shape)?)
        .map(Array::from)
        .map_err(TCError::from)
}
