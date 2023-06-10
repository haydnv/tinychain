use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use b_table::b_tree::Collator;
use b_table::{TableLock, TableWriteGuard};
use freqfs::DirLock;
use futures::future::TryFutureExt;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use futures::try_join;
use ha_ndarray::*;
use itertools::Itertools;
use rayon::prelude::*;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::TxnId;
use tc_value::{DType, Number, NumberClass, NumberCollator, NumberType};
use tcgeneric::ThreadSafe;

use crate::tensor::transform::{Expand, Reshape, Slice, Transpose};
use crate::tensor::{
    coord_of, offset_of, strides_for, validate_order, Axes, AxisRange, Coord, Range, Semaphore,
    Shape, TensorInstance, TensorPermitRead, TensorPermitWrite,
};

use super::schema::{IndexSchema, Schema};
use super::{stream, Blocks, Elements, Node, SparseInstance};

#[async_trait]
pub trait SparseWriteLock<'a>: SparseInstance {
    type Guard: SparseWriteGuard<Self::DType>;

    async fn write(&'a self) -> Self::Guard;
}

#[async_trait]
pub trait SparseWriteGuard<T: CDatatype + DType>: Send + Sync {
    async fn clear(&mut self, range: Range) -> TCResult<()>;

    async fn merge<FE>(
        &mut self,
        filled: SparseFile<FE, T>,
        zeros: SparseFile<FE, T>,
    ) -> TCResult<()>
    where
        FE: AsType<Node> + ThreadSafe,
        Number: CastInto<T>,
    {
        let mut zeros = zeros
            .table
            .rows(b_table::Range::default(), &[], false)
            .await?;

        while let Some(row) = zeros.try_next().await? {
            let (coord, zero) = unwrap_row(row);
            self.write_value(coord, zero).await?;
        }

        let mut filled = filled
            .table
            .rows(b_table::Range::default(), &[], false)
            .await?;

        while let Some(row) = filled.try_next().await? {
            let (coord, value) = unwrap_row(row);
            self.write_value(coord, value).await?;
        }

        Ok(())
    }

    async fn overwrite<O: SparseInstance<DType = T>>(&mut self, other: O) -> TCResult<()>;

    async fn write_value(&mut self, coord: Coord, value: T) -> TCResult<()>;
}

pub enum SparseAccessCast<FE> {
    F32(SparseAccess<FE, f32>),
    F64(SparseAccess<FE, f64>),
    I16(SparseAccess<FE, i16>),
    I32(SparseAccess<FE, i32>),
    I64(SparseAccess<FE, i64>),
    U8(SparseAccess<FE, u8>),
    U16(SparseAccess<FE, u16>),
    U32(SparseAccess<FE, u32>),
    U64(SparseAccess<FE, u64>),
}

impl<FE> Clone for SparseAccessCast<FE> {
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

impl<FE: Send + Sync + 'static> SparseAccessCast<FE> {
    pub fn dtype(&self) -> NumberType {
        access_cast_dispatch!(self, this, this.dtype())
    }

    pub fn shape(&self) -> &Shape {
        access_cast_dispatch!(self, this, this.shape())
    }
}

impl<FE> SparseAccessCast<FE>
where
    FE: AsType<Node> + ThreadSafe,
{
    pub async fn read_value(&self, coord: Coord) -> TCResult<Number> {
        access_cast_dispatch!(
            self,
            this,
            this.read_value(coord).map_ok(Number::from).await
        )
    }

    pub async fn elements(self, range: Range, order: Axes) -> TCResult<Elements<Number>> {
        access_cast_dispatch!(self, this, {
            let elements = this.elements(range, order).await?;
            Ok(Box::pin(
                elements.map_ok(|(coord, value)| (coord, Number::from(value))),
            ))
        })
    }

    pub async fn inner_join(
        self,
        other: Self,
        range: Range,
        order: Axes,
    ) -> TCResult<Elements<(Number, Number)>> {
        let shape = self.shape().to_vec();
        let strides = strides_for(&shape, shape.len());

        let this_shape = self.shape().to_vec();
        let that_shape = other.shape().to_vec();

        let (this, that) = try_join!(
            self.elements(range.clone(), order.to_vec()),
            other.elements(range, order)
        )?;

        let this = this.map_ok(move |(coord, value)| (offset_of(coord, &this_shape), value));
        let that = that.map_ok(move |(coord, value)| (offset_of(coord, &that_shape), value));

        let elements = stream::InnerJoin::new(this, that).map_ok(move |(offset, left, right)| {
            let coord = coord_of(offset, &strides, &shape);
            (coord, (left, right))
        });

        Ok(Box::pin(elements))
    }

    pub async fn outer_join(
        self,
        other: Self,
        range: Range,
        order: Axes,
    ) -> TCResult<Elements<(Number, Number)>> {
        let zero = self.dtype().zero();
        let shape = self.shape().to_vec();
        let strides = strides_for(&shape, shape.len());

        let this_shape = self.shape().to_vec();
        let that_shape = other.shape().to_vec();

        let (this, that) = try_join!(
            self.elements(range.clone(), order.to_vec()),
            other.elements(range, order)
        )?;

        let this = this.map_ok(move |(coord, value)| (offset_of(coord, &this_shape), value));
        let that = that.map_ok(move |(coord, value)| (offset_of(coord, &that_shape), value));

        let elements =
            stream::OuterJoin::new(this, that, zero).map_ok(move |(offset, left, right)| {
                let coord = coord_of(offset, &strides, &shape);
                (coord, (left, right))
            });

        Ok(Box::pin(elements))
    }
}

#[async_trait]
impl<FE: Send + Sync + 'static> TensorPermitRead for SparseAccessCast<FE> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        access_cast_dispatch!(self, this, this.read_permit(txn_id, range).await)
    }
}

macro_rules! access_cast_from {
    ($t:ty, $var:ident) => {
        impl<FE> From<SparseAccess<FE, $t>> for SparseAccessCast<FE> {
            fn from(access: SparseAccess<FE, $t>) -> Self {
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

impl<FE> fmt::Debug for SparseAccessCast<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        access_cast_dispatch!(self, this, this.fmt(f))
    }
}

pub enum SparseAccess<FE, T> {
    Table(SparseFile<FE, T>),
    Broadcast(Box<SparseBroadcast<FE, T>>),
    BroadcastAxis(Box<SparseBroadcastAxis<Self>>),
    Cast(Box<SparseCast<FE, T>>),
    Combine(Box<SparseCombine<FE, T>>),
    CombineLeft(Box<SparseLeftCombine<FE, T>>),
    Cow(Box<SparseCow<FE, T, Self>>),
    Expand(Box<SparseExpand<Self>>),
    Reshape(Box<SparseReshape<Self>>),
    Slice(Box<SparseSlice<Self>>),
    Transpose(Box<SparseTranspose<Self>>),
    Version(SparseVersion<FE, T>),
}

impl<FE, T> Clone for SparseAccess<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Table(table) => Self::Table(table.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::BroadcastAxis(broadcast) => Self::BroadcastAxis(broadcast.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
            Self::Combine(combine) => Self::Combine(combine.clone()),
            Self::CombineLeft(combine) => Self::CombineLeft(combine.clone()),
            Self::Cow(cow) => Self::Cow(cow.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
            Self::Version(version) => Self::Version(version.clone()),
        }
    }
}

macro_rules! access_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Table($var) => $call,
            Self::Broadcast($var) => $call,
            Self::BroadcastAxis($var) => $call,
            Self::Cast($var) => $call,
            Self::Combine($var) => $call,
            Self::CombineLeft($var) => $call,
            Self::Cow($var) => $call,
            Self::Expand($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
            Self::Transpose($var) => $call,
            Self::Version($var) => $call,
        }
    };
}

impl<FE, T> TensorInstance for SparseAccess<FE, T>
where
    FE: Send + Sync + 'static,
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
impl<FE, T> SparseInstance for SparseAccess<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Array<u64>, Array<T>>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        match self {
            Self::Table(table) => {
                let blocks = table.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Broadcast(broadcast) => {
                let blocks = broadcast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::BroadcastAxis(broadcast) => {
                let blocks = broadcast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Cast(cast) => {
                let blocks = cast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Combine(combine) => {
                let blocks = combine.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::CombineLeft(combine) => {
                let blocks = combine.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Cow(cow) => {
                let blocks = cow.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Expand(expand) => {
                let blocks = expand.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Reshape(reshape) => {
                let blocks = reshape.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Slice(slice) => {
                let blocks = slice.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Transpose(transpose) => {
                let blocks = transpose.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Version(version) => {
                let blocks = version.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
        }
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        access_dispatch!(self, this, this.elements(range, order).await)
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        access_dispatch!(self, this, this.read_value(coord).await)
    }
}

#[async_trait]
impl<FE, T> TensorPermitRead for SparseAccess<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::BroadcastAxis(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::Cow(cow) => cow.read_permit(txn_id, range).await,
            Self::Combine(combine) => combine.read_permit(txn_id, range).await,
            Self::CombineLeft(combine) => combine.read_permit(txn_id, range).await,
            Self::Expand(expand) => expand.read_permit(txn_id, range).await,
            Self::Reshape(reshape) => reshape.read_permit(txn_id, range).await,
            Self::Slice(slice) => slice.read_permit(txn_id, range).await,
            Self::Transpose(transpose) => transpose.read_permit(txn_id, range).await,
            Self::Version(version) => version.read_permit(txn_id, range).await,
            other => Err(bad_request!(
                "{:?} does not support transactional reads",
                other
            )),
        }
    }
}

#[async_trait]
impl<FE, T> TensorPermitWrite for SparseAccess<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        match self {
            Self::Slice(slice) => slice.write_permit(txn_id, range).await,
            Self::Version(version) => version.write_permit(txn_id, range).await,
            other => Err(bad_request!(
                "{:?} does not support transactional writes",
                other
            )),
        }
    }
}

impl<FE, T: DType> fmt::Debug for SparseAccess<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        access_dispatch!(self, this, this.fmt(f))
    }
}

pub struct SparseFile<FE, T> {
    table: TableLock<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for SparseFile<FE, T> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE, T> SparseFile<FE, T> {
    pub fn collator(&self) -> &Arc<Collator<NumberCollator>> {
        self.table.collator()
    }

    pub fn schema(&self) -> &Schema {
        self.table.schema()
    }
}

impl<FE: AsType<Node> + Send + Sync, T> SparseFile<FE, T> {
    pub fn create(dir: DirLock<FE>, shape: Shape) -> Result<Self, TCError> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::create(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }

    pub fn load(dir: DirLock<FE>, shape: Shape) -> Result<Self, TCError> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::load(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for SparseFile<FE, T>
where
    FE: Send + Sync + 'static,
    T: DType + ThreadSafe,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.table.schema().shape()
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseFile<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
    Number: CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<T>, T>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        self.shape().validate_range(&range)?;
        debug_assert!(validate_order(&order, self.ndim()));

        let range = table_range(&range)?;
        let rows = self.table.rows(range, &order, false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(TCError::from);
        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;

        let key = coord.into_iter().map(Number::from).collect();
        let table = self.table.read().await;
        if let Some(mut row) = table.get(&key).await? {
            let value = row.pop().expect("value");
            Ok(value.cast_into())
        } else {
            Ok(T::zero())
        }
    }
}

#[async_trait]
impl<'a, FE, T> SparseWriteLock<'a> for SparseFile<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Guard = SparseTableWriteGuard<'a, FE, T>;

    async fn write(&'a self) -> SparseTableWriteGuard<'a, FE, T> {
        SparseTableWriteGuard {
            shape: self.table.schema().shape(),
            table: self.table.write().await,
            dtype: self.dtype,
        }
    }
}

impl<FE, T> From<SparseFile<FE, T>> for SparseAccess<FE, T> {
    fn from(table: SparseFile<FE, T>) -> Self {
        Self::Table(table)
    }
}

impl<FE, T> fmt::Debug for SparseFile<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "sparse table with shape {:?}",
            self.table.schema().shape()
        )
    }
}

pub struct SparseTableWriteGuard<'a, FE, T> {
    shape: &'a Shape,
    table: TableWriteGuard<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

#[async_trait]
impl<'a, FE, T> SparseWriteGuard<T> for SparseTableWriteGuard<'a, FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T>,
{
    async fn clear(&mut self, range: Range) -> TCResult<()> {
        if range == Range::default() || range == Range::all(&self.shape) {
            self.table.truncate().map_err(TCError::from).await
        } else {
            Err(not_implemented!("delete {range:?}"))
        }
    }

    async fn overwrite<O: SparseInstance<DType = T>>(&mut self, other: O) -> TCResult<()> {
        if self.shape != other.shape() {
            return Err(bad_request!(
                "cannot overwrite a tensor of shape {:?} with {:?}",
                self.shape,
                other.shape()
            ));
        }

        self.clear(Range::default()).await?;

        let order = (0..other.ndim()).into_iter().collect();
        let mut elements = other.elements(Range::default(), order).await?;

        while let Some((coord, value)) = elements.try_next().await? {
            let coord = coord.into_iter().map(|i| Number::UInt(i.into())).collect();
            self.table.upsert(coord, vec![value.into()]).await?;
        }

        Ok(())
    }

    async fn write_value(&mut self, coord: Coord, value: T) -> Result<(), TCError> {
        self.shape.validate_coord(&coord)?;

        let coord = coord.into_iter().map(|i| Number::UInt(i.into())).collect();

        if value == T::zero() {
            self.table.delete(&coord).await?;
        } else {
            self.table.upsert(coord, vec![value.into()]).await?;
        }

        Ok(())
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
    FE: Send + Sync,
    T: CDatatype + DType,
    SparseFile<FE, T>: TensorInstance,
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
    FE: Send + Sync,
    T: CDatatype + DType,
    SparseFile<FE, T>: SparseInstance,
{
    type CoordBlock = <SparseFile<FE, T> as SparseInstance>::CoordBlock;
    type ValueBlock = <SparseFile<FE, T> as SparseInstance>::ValueBlock;
    type Blocks = <SparseFile<FE, T> as SparseInstance>::Blocks;
    type DType = <SparseFile<FE, T> as SparseInstance>::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        self.file.blocks(range, order).await
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        self.file.elements(range, order).await
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.file.read_value(coord).await
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
    FE: Send + Sync,
    T: CDatatype + DType,
    SparseFile<FE, T>: SparseWriteLock<'a>,
{
    type Guard = <SparseFile<FE, T> as SparseWriteLock<'a>>::Guard;

    async fn write(&'a self) -> Self::Guard {
        self.file.write().await
    }
}

impl<FE, T> From<SparseVersion<FE, T>> for SparseAccess<FE, T> {
    fn from(version: SparseVersion<FE, T>) -> Self {
        Self::Version(version)
    }
}

impl<FE, T> fmt::Debug for SparseVersion<FE, T>
where
    SparseFile<FE, T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional version of {:?}", self.file)
    }
}

pub struct SparseBroadcast<FE, T> {
    shape: Shape,
    inner: SparseAccess<FE, T>,
}

impl<FE, T> Clone for SparseBroadcast<FE, T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<FE, T> SparseBroadcast<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
{
    pub fn new<S>(source: S, shape: Shape) -> Result<Self, TCError>
    where
        S: TensorInstance + Into<SparseAccess<FE, T>>,
    {
        let source_shape = source.shape().to_vec();
        let mut inner = source.into();

        let axes = (0..source_shape.len()).into_iter().rev();
        let dims = source_shape
            .into_iter()
            .rev()
            .zip(shape.iter().rev().copied());

        for (x, (dim, bdim)) in axes.zip(dims) {
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

        Ok(Self { shape, inner })
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseBroadcast<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseBroadcast<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        mut range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let offset = ndim - self.inner.ndim();

        if offset == 0 {
            return self.inner.elements(range, order).await;
        }

        self.shape.validate_range(&range)?;
        debug_assert!(validate_order(&order, ndim));

        let mut inner_range = Vec::with_capacity(self.inner.ndim());
        while range.len() > offset {
            inner_range.push(range.pop().expect("bound"));
        }

        let inner_order = if order
            .iter()
            .take(offset)
            .copied()
            .enumerate()
            .all(|(o, x)| x == o)
        {
            Ok(order.iter().skip(offset).cloned().collect::<Axes>())
        } else {
            Err(bad_request!(
                "an outer broadcast of a sparse tensor does not support permutation"
            ))
        }?;

        let outer = range.into_iter().multi_cartesian_product();

        let inner = self.inner;
        let elements = futures::stream::iter(outer)
            .then(move |outer_coord| {
                let inner = inner.clone();
                let inner_range = inner_range.to_vec();
                let inner_order = inner_order.to_vec();

                async move {
                    let inner_elements = inner.elements(inner_range.into(), inner_order).await?;

                    let elements = inner_elements.map_ok(move |(inner_coord, value)| {
                        let mut coord = Vec::with_capacity(ndim);
                        coord.extend_from_slice(&outer_coord);
                        coord.extend(inner_coord);
                        (coord, value)
                    });

                    Result::<_, TCError>::Ok(elements)
                }
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, mut coord: Coord) -> Result<Self::DType, TCError> {
        while coord.len() > self.inner.ndim() {
            coord.remove(0);
        }

        self.inner.read_value(coord).await
    }
}

#[async_trait]
impl<FE, T> TensorPermitRead for SparseBroadcast<FE, T>
where
    SparseAccess<FE, T>: TensorPermitRead,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        mut range: Range,
    ) -> TCResult<Vec<PermitRead<Range>>> {
        self.shape.validate_range(&range)?;

        while range.len() > self.shape.len() {
            range.remove(0);
        }

        self.inner.read_permit(txn_id, range).await
    }
}

impl<FE, T> From<SparseBroadcast<FE, T>> for SparseAccess<FE, T> {
    fn from(accessor: SparseBroadcast<FE, T>) -> Self {
        Self::Broadcast(Box::new(accessor))
    }
}

impl<FE, T> fmt::Debug for SparseBroadcast<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "broadcasted sparse tensor with shape {:?}", self.shape)
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
    fn new(source: S, axis: usize, dim: u64) -> Result<Self, TCError> {
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

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        range: Range,
        mut order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        self.shape.validate_range(&range)?;
        debug_assert!(validate_order(&order, self.ndim()));

        let axis = self.axis;
        let ndim = self.shape.len();

        let (source_range, dim) = if range.len() > axis {
            let bdim = match &range[axis] {
                AxisRange::At(i) if *i < self.dim => Ok(1),
                AxisRange::In(axis_range, 1) if axis_range.end > axis_range.start => {
                    Ok(axis_range.end - axis_range.start)
                }
                bound => Err(bad_request!(
                    "invalid bound for axis {}: {:?}",
                    self.axis,
                    bound
                )),
            }?;

            let mut source_range = range;
            source_range[axis] = AxisRange::At(0);
            (source_range, bdim)
        } else {
            (range, self.dim)
        };

        let (source_order, inner_order) = if order
            .iter()
            .take(axis)
            .copied()
            .enumerate()
            .all(|(o, x)| x == o)
        {
            let mut inner_order = Axes::with_capacity(ndim - axis);

            while order.len() > axis {
                inner_order.push(order.pop().expect("axis"));
            }

            Ok((order, inner_order))
        } else {
            Err(bad_request!(
                "an outer broadcast of a sparse tensor does not support permutation"
            ))
        }?;

        if self.axis == self.ndim() - 1 {
            let source_elements = self.source.elements(source_range, source_order).await?;

            // TODO: write a range to a slice of a coordinate block instead
            let elements = source_elements
                .map_ok(move |(source_coord, value)| {
                    futures::stream::iter(0..dim).map(move |i| {
                        let mut coord = source_coord.to_vec();
                        *coord.last_mut().expect("x") = i;

                        Ok((coord, value))
                    })
                })
                .try_flatten();

            Ok(Box::pin(elements))
        } else {
            let axes = (0..axis).into_iter().collect();
            let inner_range = source_range.iter().skip(axis).cloned().collect::<Vec<_>>();

            let source = self.source;
            let filled = source.clone().filled_at(source_range, axes).await?;

            let elements = filled
                .map(move |result| {
                    let outer_coord = result?;
                    debug_assert_eq!(outer_coord.len(), axis);
                    debug_assert_eq!(outer_coord.last(), Some(&0));

                    let inner_range = inner_range.to_vec();
                    let inner_order = inner_order.to_vec();

                    let prefix = outer_coord
                        .iter()
                        .copied()
                        .map(|i| AxisRange::At(i))
                        .collect();

                    let slice = SparseSlice::new(source.clone(), prefix)?;

                    let elements = futures::stream::iter(0..dim)
                        .then(move |i| {
                            let outer_coord = outer_coord[..axis - 1].to_vec();
                            let inner_range = inner_range.to_vec().into();
                            let inner_order = inner_order.to_vec();
                            let slice = slice.clone();

                            async move {
                                let inner_elements =
                                    slice.elements(inner_range, inner_order).await?;

                                let elements =
                                    inner_elements.map_ok(move |(inner_coord, value)| {
                                        let mut coord = Coord::with_capacity(ndim);
                                        coord.copy_from_slice(&outer_coord);
                                        coord.push(i);
                                        coord.extend(inner_coord);

                                        (coord, value)
                                    });

                                Result::<_, TCError>::Ok(elements)
                            }
                        })
                        .try_flatten();

                    Result::<_, TCError>::Ok(elements)
                })
                .try_flatten();

            Ok(Box::pin(elements))
        }
    }

    async fn read_value(&self, mut coord: Coord) -> Result<Self::DType, TCError> {
        self.shape.validate_coord(&coord)?;
        coord[self.axis] = 0;
        self.source.read_value(coord).await
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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseBroadcastAxis<S>> for SparseAccess<FE, T> {
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

pub enum SparseCastSource<FE> {
    F32(SparseAccess<FE, f32>),
    F64(SparseAccess<FE, f64>),
    I16(SparseAccess<FE, i16>),
    I32(SparseAccess<FE, i32>),
    I64(SparseAccess<FE, i64>),
    U8(SparseAccess<FE, u8>),
    U16(SparseAccess<FE, u16>),
    U32(SparseAccess<FE, u32>),
    U64(SparseAccess<FE, u64>),
}

impl<FE> Clone for SparseCastSource<FE> {
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

macro_rules! cast_source {
    ($t:ty, $var:ident) => {
        impl<FE> From<SparseAccess<FE, $t>> for SparseCastSource<FE> {
            fn from(access: SparseAccess<FE, $t>) -> Self {
                Self::$var(access)
            }
        }
    };
}

cast_source!(f32, F32);
cast_source!(f64, F64);
cast_source!(i16, I16);
cast_source!(i32, I32);
cast_source!(i64, I64);
cast_source!(u8, U8);
cast_source!(u16, U16);
cast_source!(u32, U32);
cast_source!(u64, U64);

macro_rules! cast_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            SparseCastSource::F32($var) => $call,
            SparseCastSource::F64($var) => $call,
            SparseCastSource::I16($var) => $call,
            SparseCastSource::I32($var) => $call,
            SparseCastSource::I64($var) => $call,
            SparseCastSource::U8($var) => $call,
            SparseCastSource::U16($var) => $call,
            SparseCastSource::U32($var) => $call,
            SparseCastSource::U64($var) => $call,
        }
    };
}

impl<FE> fmt::Debug for SparseCastSource<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        cast_dispatch!(self, this, this.fmt(f))
    }
}

pub struct SparseCast<FE, T> {
    source: SparseCastSource<FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for SparseCast<FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            dtype: self.dtype,
        }
    }
}

impl<FE, T> TensorInstance for SparseCast<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let source = &self.source;
        cast_dispatch!(source, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseCast<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        match self.source {
            SparseCastSource::F32(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::F64(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::I16(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::I32(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::I64(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U8(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U16(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U32(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U64(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
        }
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.read(&queue)?.to_slice()?;
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

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        let source = &self.source;
        cast_dispatch!(source, this, {
            let value: Number = this.read_value(coord).map_ok(|n| n.into()).await?;
            Ok(value.cast_into())
        })
    }
}

impl<FE, T: DType> fmt::Debug for SparseCast<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cast {:?} into {:?}", self.source, T::dtype())
    }
}

pub struct SparseCombine<FE, T> {
    left: SparseAccessCast<FE>,
    right: SparseAccessCast<FE>,
    op: fn(Number, Number) -> T,
}

impl<FE, T> Clone for SparseCombine<FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            op: self.op,
        }
    }
}

impl<FE, T> SparseCombine<FE, T>
where
    FE: Send + Sync + 'static,
{
    pub fn new<L, R>(left: L, right: R, op: fn(Number, Number) -> T) -> TCResult<Self>
    where
        L: Into<SparseAccessCast<FE>>,
        R: Into<SparseAccessCast<FE>>,
    {
        let left = left.into();
        let right = right.into();

        if left.shape() == right.shape() {
            Ok(Self { left, right, op })
        } else {
            Err(bad_request!(
                "cannot combine {:?} with {:?} (wrong shape)",
                left,
                right
            ))
        }
    }
}

impl<FE, T> TensorInstance for SparseCombine<FE, T>
where
    FE: Send + Sync + 'static,
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
impl<FE, T> SparseInstance for SparseCombine<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, T>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let elements = self.left.outer_join(self.right, range, order).await?;
        Ok(Box::pin(elements.map_ok(move |(coord, (l, r))| {
            let value = (self.op)(l, r);
            (coord, value)
        })))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        let (left, right) = try_join!(
            self.left.read_value(coord.to_vec()),
            self.right.read_value(coord)
        )?;

        Ok((self.op)(left, right))
    }
}

#[async_trait]
impl<FE: Send + Sync + 'static, T> TensorPermitRead for SparseCombine<FE, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<FE, T> fmt::Debug for SparseCombine<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} and {:?}", self.left, self.right)
    }
}

pub struct SparseLeftCombine<FE, T> {
    left: SparseAccessCast<FE>,
    right: SparseAccessCast<FE>,
    op: fn(Number, Number) -> T,
}

impl<FE, T> Clone for SparseLeftCombine<FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            op: self.op,
        }
    }
}

impl<FE, T> SparseLeftCombine<FE, T>
where
    FE: Send + Sync + 'static,
{
    pub fn new<L, R>(left: L, right: R, op: fn(Number, Number) -> T) -> TCResult<Self>
    where
        L: Into<SparseAccessCast<FE>>,
        R: Into<SparseAccessCast<FE>>,
    {
        let left = left.into();
        let right = right.into();

        if left.shape() == right.shape() {
            Ok(Self { left, right, op })
        } else {
            Err(bad_request!(
                "cannot combine {:?} with {:?} (wrong shape)",
                left,
                right
            ))
        }
    }
}

impl<FE, T> TensorInstance for SparseLeftCombine<FE, T>
where
    FE: Send + Sync + 'static,
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
impl<FE, T> SparseInstance for SparseLeftCombine<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, T>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let elements = self.left.inner_join(self.right, range, order).await?;
        Ok(Box::pin(elements.map_ok(move |(coord, (l, r))| {
            let value = (self.op)(l, r);
            (coord, value)
        })))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        let (left, right) = try_join!(
            self.left.read_value(coord.to_vec()),
            self.right.read_value(coord)
        )?;

        Ok((self.op)(left, right))
    }
}

#[async_trait]
impl<FE: Send + Sync + 'static, T> TensorPermitRead for SparseLeftCombine<FE, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these locks in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range.clone()).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<FE, T> fmt::Debug for SparseLeftCombine<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} and {:?}", self.left, self.right)
    }
}

pub struct SparseCombineConst<FE, T> {
    left: SparseAccessCast<FE>,
    right: Number,
    op: fn(Number, Number) -> T,
}

impl<FE, T> Clone for SparseCombineConst<FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right,
            op: self.op,
        }
    }
}

impl<FE, T> SparseCombineConst<FE, T> {
    pub fn new<L>(left: L, right: Number, op: fn(Number, Number) -> T) -> Self
    where
        L: Into<SparseAccessCast<FE>>,
    {
        Self {
            left: left.into(),
            right,
            op,
        }
    }
}

impl<FE: ThreadSafe, T: CDatatype + DType> TensorInstance for SparseCombineConst<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.left.shape()
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseCombineConst<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = stream::BlockCoords<Elements<T>, T>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let left_elements = self.left.elements(range, order).await?;

        let elements = left_elements.map_ok(move |(coord, l)| {
            let value = (self.op)(l, self.right);
            (coord, value)
        });

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.left
            .read_value(coord)
            .map_ok(move |l| (self.op)(l, self.right))
            .await
    }
}

#[async_trait]
impl<FE: ThreadSafe, T: CDatatype> TensorPermitRead for SparseCombineConst<FE, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<FE, T> fmt::Debug for SparseCombineConst<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with {:?}", self.left, self.right)
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
    pub fn create(source: S, filled: SparseFile<FE, T>, zeros: SparseFile<FE, T>) -> Self {
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
    FE: Send + Sync + 'static,
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
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let shape = self.source.shape().to_vec();
        let ndim = shape.len();
        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context.clone(), size_hint(self.size()))?;

        let strides = strides_for(&shape, ndim);
        let strides = ArrayBase::<Arc<Vec<_>>>::with_context(
            context.clone(),
            vec![strides.len()],
            Arc::new(strides),
        )?;

        let (source_blocks, filled_blocks, zero_blocks) = try_join!(
            self.source.blocks(range.clone(), order.to_vec()),
            self.filled.blocks(range.clone(), order.to_vec()),
            self.zeros.blocks(range, order)
        )?;

        let source_elements = offsets(queue.clone(), strides.clone(), source_blocks);
        let filled_elements = offsets(queue.clone(), strides.clone(), filled_blocks);
        let zero_elements = offsets(queue, strides.clone(), zero_blocks);

        let elements = stream::TryDiff::new(source_elements, zero_elements);
        let elements = stream::TryMerge::new(elements, filled_elements);
        let offsets = stream::BlockOffsets::new(elements);

        let dims = ArrayBase::<Arc<Vec<_>>>::with_context(context, vec![ndim], Arc::new(shape))?;
        let blocks = offsets.map(move |result| {
            let (offsets, values) = result?;

            let num_offsets = offsets.size();
            let block_shape = vec![num_offsets, ndim];
            let coords = offsets
                .expand_dims(vec![1])?
                .broadcast(block_shape.to_vec())?
                .mul(strides.clone().broadcast(block_shape.to_vec())?)?
                .rem(dims.clone().broadcast(block_shape)?)?;

            let coords = ArrayBase::<Vec<_>>::copy(&coords)?;

            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(range, order).await?;
        let elements = blocks
            .map_ok(move |(coords, values)| {
                let tuples = coords
                    .into_inner()
                    .into_par_iter()
                    .chunks(ndim)
                    .zip(values.into_inner());

                futures::stream::iter(tuples.map(Ok).collect::<Vec<_>>())
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;

        let key = coord.iter().copied().map(Number::from).collect();

        {
            let zeros = self.zeros.table.read().await;
            if zeros.contains(&key).await? {
                return Ok(Self::DType::zero());
            }
        }

        {
            let filled = self.filled.table.read().await;
            if let Some(mut row) = filled.get(&key).await? {
                let value = row.pop().expect("value");
                return Ok(value.cast_into());
            }
        }

        self.source.read_value(coord).await
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
    type Guard = SparseCowWriteGuard<'a, FE, Self, T>;

    async fn write(&'a self) -> Self::Guard {
        SparseCowWriteGuard {
            source: self,
            filled: self.filled.write().await,
            zeros: self.zeros.write().await,
        }
    }
}

impl<FE, T, S> From<SparseCow<FE, T, S>> for SparseAccess<FE, T>
where
    S: Into<SparseAccess<FE, T>>,
{
    fn from(cow: SparseCow<FE, T, S>) -> Self {
        SparseAccess::Cow(Box::new(SparseCow {
            source: cow.source.into(),
            filled: cow.filled,
            zeros: cow.zeros,
        }))
    }
}

impl<FE, T, S: fmt::Debug> fmt::Debug for SparseCow<FE, T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "copy-on-write view of {:?}", self.source)
    }
}

pub struct SparseCowWriteGuard<'a, FE, S, T> {
    source: &'a S,
    filled: SparseTableWriteGuard<'a, FE, T>,
    zeros: SparseTableWriteGuard<'a, FE, T>,
}

#[async_trait]
impl<'a, FE, S, T> SparseWriteGuard<T> for SparseCowWriteGuard<'a, FE, S, T>
where
    FE: AsType<Node> + ThreadSafe,
    S: SparseInstance<DType = T> + Clone,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T>,
{
    async fn clear(&mut self, range: Range) -> TCResult<()> {
        self.filled.clear(range.clone()).await?;
        self.zeros.clear(range.clone()).await?;

        let order = (0..self.source.ndim()).into_iter().collect();
        let mut elements = self.source.clone().elements(range, order).await?;

        while let Some((coord, value)) = elements.try_next().await? {
            self.zeros.write_value(coord, value).await?;
        }

        Ok(())
    }

    async fn overwrite<O: SparseInstance<DType = T>>(&mut self, other: O) -> TCResult<()> {
        if self.source.shape() != other.shape() {
            return Err(bad_request!(
                "cannot overwrite a sparse tensor of shape {:?} with one of shape {:?}",
                self.source.shape(),
                other.shape()
            ));
        }

        self.clear(Range::default()).await?;

        let order = (0..other.ndim()).into_iter().collect();
        let mut elements = other.elements(Range::default(), order).await?;

        while let Some((coord, value)) = elements.try_next().await? {
            self.write_value(coord, value).await?;
        }

        Ok(())
    }

    async fn write_value(&mut self, coord: Coord, value: T) -> Result<(), TCError> {
        let inverse = if value == T::zero() {
            T::one()
        } else {
            T::zero()
        };

        try_join!(
            self.filled.write_value(coord.to_vec(), value),
            self.zeros.write_value(coord, inverse)
        )
        .map(|_| ())
    }
}

#[derive(Clone)]
pub struct SparseExpand<S> {
    source: S,
    transform: Expand,
}

impl<S: TensorInstance + fmt::Debug> SparseExpand<S> {
    pub fn new(source: S, axes: Axes) -> Result<Self, TCError> {
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

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        self.shape().validate_range(&range)?;
        debug_assert!(validate_order(&order, self.ndim()));

        let mut source_range = range;
        for x in self.transform.expand_axes().iter().rev().copied() {
            if x < source_range.len() {
                source_range.remove(x);
            }
        }

        let mut source_order = order;
        for x in self.transform.expand_axes().iter().rev().copied() {
            source_order.remove(x);
        }

        let ndim = self.ndim();
        let axes = self.transform.expand_axes().to_vec();
        debug_assert_eq!(self.source.ndim() + 1, ndim);

        let source_elements = self.source.elements(source_range, source_order).await?;

        let elements = source_elements.map_ok(move |(source_coord, value)| {
            let mut coord = Coord::with_capacity(ndim);
            coord.extend(source_coord);
            for x in axes.iter().rev().copied() {
                coord.insert(x, 0);
            }

            debug_assert_eq!(coord.len(), ndim);
            (coord, value)
        });

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(source_coord).await
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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseExpand<S>> for SparseAccess<FE, T> {
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
    pub fn new(source: S, shape: Shape) -> Result<Self, TCError> {
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
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        self.shape().validate_range(&range)?;
        debug_assert!(validate_order(&order, self.ndim()));

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
        let source_blocks = self.source.blocks(source_range, source_order).await?;
        let source_strides = Arc::new(self.transform.source_strides().to_vec());
        let source_strides = ArrayBase::<Arc<Vec<_>>>::new(vec![source_ndim], source_strides)?;

        let ndim = self.transform.shape().len();
        let strides = Arc::new(self.transform.strides().to_vec());
        let strides = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], strides)?;
        let shape = Arc::new(self.transform.shape().to_vec());
        let shape = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], shape)?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;

            debug_assert_eq!(source_coords.size() % source_ndim, 0);
            debug_assert_eq!(source_coords.size() / source_ndim, values.size());

            let source_strides = source_strides
                .clone()
                .broadcast(vec![values.size(), source_ndim])?;

            let offsets = source_coords.mul(source_strides)?;
            let offsets = offsets.sum_axis(1, false)?;

            let broadcast = vec![offsets.size(), ndim];
            let strides = strides.clone().broadcast(broadcast.to_vec())?;
            let offsets = offsets
                .expand_dims(vec![1])?
                .broadcast(broadcast.to_vec())?;

            let dims = shape.clone().expand_dims(vec![0])?.broadcast(broadcast)?;
            let coords = (offsets / strides) % dims;

            let coords = ArrayBase::<Vec<_>>::copy(&coords)?;

            Result::<_, TCError>::Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.into_inner();
                let values = values.read(&queue)?.to_slice()?;
                let tuples = coords
                    .into_par_iter()
                    .chunks(ndim)
                    .zip(values.as_ref().into_par_iter().copied())
                    .map(Ok)
                    .collect::<Vec<_>>();

                Result::<_, TCError>::Ok(futures::stream::iter(tuples))
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(source_coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for SparseReshape<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        if range.is_empty() || range == Range::all(self.transform.shape()) {
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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseReshape<S>> for SparseAccess<FE, T> {
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
    pub fn new(source: S, range: Range) -> Result<Self, TCError> {
        Slice::new(source.shape().clone(), range).map(|transform| Self { source, transform })
    }

    fn source_order(&self, order: Axes) -> Result<Axes, TCError> {
        debug_assert!(validate_order(&order, self.ndim()));

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
    type CoordBlock = S::CoordBlock;
    type ValueBlock = S::ValueBlock;
    type Blocks = S::Blocks;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        self.shape().validate_range(&range)?;

        let source_order = self.source_order(order)?;
        let source_range = self.transform.invert_range(range);

        self.source.blocks(source_range, source_order).await
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        self.shape().validate_range(&range)?;

        let source_order = self.source_order(order)?;
        let source_range = self.transform.invert_range(range);

        self.source.elements(source_range, source_order).await
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord)?;
        self.source.read_value(source_coord).await
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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseSlice<S>> for SparseAccess<FE, T> {
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
            "slice of {:?} with range {:?}",
            self.source,
            self.transform.range()
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
    async fn clear(&mut self, range: Range) -> TCResult<()> {
        self.transform.shape().validate_range(&range)?;
        self.guard.clear(self.transform.invert_range(range)).await
    }

    async fn overwrite<O: SparseInstance<DType = T>>(&mut self, other: O) -> TCResult<()> {
        if self.transform.shape() != other.shape() {
            return Err(bad_request!(
                "cannot overwrite a sparse tensor of shape {:?} with one of shape {:?}",
                self.transform.shape(),
                other.shape()
            ));
        }

        self.clear(Range::default()).await?;

        let order = (0..other.shape().len()).into_iter().collect();
        let mut elements = other.elements(Range::default(), order).await?;

        while let Some((coord, value)) = elements.try_next().await? {
            self.write_value(coord, value).await?;
        }

        Ok(())
    }

    async fn write_value(&mut self, coord: Coord, value: T) -> Result<(), TCError> {
        self.transform.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord)?;
        self.guard.write_value(coord, value).await
    }
}

#[derive(Clone)]
pub struct SparseTranspose<S> {
    source: S,
    transform: Transpose,
}

impl<S: SparseInstance> SparseTranspose<S> {
    pub fn new(source: S, permutation: Option<Axes>) -> Result<Self, TCError> {
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
    <S::CoordBlock as NDArrayTransform>::Transpose: NDArrayRead<DType = u64>,
{
    type CoordBlock = <S::CoordBlock as NDArrayTransform>::Transpose;
    type ValueBlock = S::ValueBlock;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        self.shape().validate_range(&range)?;
        debug_assert!(validate_order(&order, self.ndim()));

        let range = range.normalize(self.shape());
        debug_assert_eq!(range.len(), self.ndim());

        let source_order = order
            .into_iter()
            .map(|x| self.transform.axes()[x])
            .collect();

        let source_range = self.transform.invert_range(&range);

        let source_blocks = self.source.blocks(source_range, source_order).await?;

        let permutation = self.transform.axes().to_vec();
        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;
            let coords = source_coords.transpose(Some(permutation.to_vec()))?;
            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.read(&queue)?.to_slice()?;
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

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;
        let source_coord = self.transform.invert_coord(coord);
        self.source.read_value(source_coord).await
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

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseTranspose<S>> for SparseAccess<FE, T> {
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

#[inline]
fn offsets<C, V, T>(
    queue: ha_ndarray::Queue,
    strides: ArrayBase<Arc<Vec<u64>>>,
    blocks: impl Stream<Item = Result<(C, V), TCError>> + Send + 'static,
) -> impl Stream<Item = Result<(u64, T), TCError>> + Send
where
    C: NDArrayRead<DType = u64> + NDArrayMath,
    V: NDArrayRead<DType = T>,
    T: CDatatype,
{
    let offsets = blocks
        .map(move |result| {
            let (coords, values) = result?;

            let strides = strides.clone().broadcast(coords.shape().to_vec())?;
            let offsets = coords.mul(strides)?.sum_axis(0, false)?;
            let offsets = offsets.read(&queue)?.to_slice()?.into_vec();

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
fn size_hint(size: u64) -> usize {
    size.try_into().ok().unwrap_or_else(|| usize::MAX)
}

#[inline]
fn unwrap_row<T>(mut row: Vec<Number>) -> (Coord, T)
where
    Number: CastInto<T> + CastInto<u64>,
{
    let n = row.pop().expect("n").cast_into();
    let coord = row.into_iter().map(|i| i.cast_into()).collect();
    (coord, n)
}

#[inline]
fn table_range(range: &Range) -> Result<b_table::Range<usize, Number>, TCError> {
    if range == &Range::default() {
        return Ok(b_table::Range::default());
    }

    let mut table_range = HashMap::new();

    for (x, bound) in range.iter().enumerate() {
        match bound {
            AxisRange::At(i) => {
                table_range.insert(x, b_table::ColumnRange::Eq(Number::from(*i)));
            }
            AxisRange::In(axis_range, 1) => {
                let start = Bound::Included(Number::from(axis_range.start));
                let stop = Bound::Excluded(Number::from(axis_range.end));
                table_range.insert(x, b_table::ColumnRange::In((start, stop)));
            }
            bound => {
                return Err(bad_request!(
                    "sparse tensor does not support axis bound {:?}",
                    bound
                ));
            }
        }
    }

    Ok(table_range.into())
}
