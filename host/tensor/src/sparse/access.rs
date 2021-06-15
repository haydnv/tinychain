use std::convert::TryFrom;

use afarray::Array;
use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, TryStreamExt};
use futures::try_join;

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::{Number, NumberClass, NumberInstance, NumberType};

use crate::stream::{sorted_values, Read, ReadValueAt};
use crate::transform::{self, Rebase};
use crate::{Coord, Shape, TensorAccess, TensorType, ERR_NONBIJECTIVE_WRITE};

use super::combine::SparseCombine;
use super::{Phantom, SparseStream, SparseTable};

#[async_trait]
pub trait SparseAccess<FD: File<Array>, FS: File<Node>, D: Dir, T: Transaction<D>>:
    Clone + ReadValueAt<D, Txn = T> + TensorAccess + Send + Sync + 'static
{
    fn accessor(self) -> SparseAccessor<FD, FS, D, T>;

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>>;

    async fn filled_count(self, txn: T) -> TCResult<u64>;

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

#[derive(Clone)]
pub enum SparseAccessor<FD, FS, D, T> {
    Broadcast(Box<SparseBroadcast<FD, FS, D, T, Self>>),
    Cast(Box<SparseCast<FD, FS, D, T, Self>>),
    Combine(Box<SparseCombinator<FD, FS, D, T, Self, Self>>),
    Expand(Box<SparseExpand<FD, FS, D, T, Self>>),
    Table(SparseTable<FD, FS, D, T>),
}

impl<FD, FS, D, T> TensorAccess for SparseAccessor<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    fn dtype(&self) -> NumberType {
        match self {
            Self::Broadcast(broadcast) => broadcast.dtype(),
            Self::Cast(cast) => cast.dtype(),
            Self::Combine(combine) => combine.dtype(),
            Self::Expand(expand) => expand.dtype(),
            Self::Table(table) => table.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Broadcast(broadcast) => broadcast.ndim(),
            Self::Cast(cast) => cast.ndim(),
            Self::Combine(combine) => combine.ndim(),
            Self::Expand(expand) => expand.ndim(),
            Self::Table(table) => table.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Broadcast(broadcast) => broadcast.shape(),
            Self::Cast(cast) => cast.shape(),
            Self::Combine(combine) => combine.shape(),
            Self::Expand(expand) => expand.shape(),
            Self::Table(table) => table.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Broadcast(broadcast) => broadcast.size(),
            Self::Cast(cast) => cast.size(),
            Self::Combine(combine) => combine.size(),
            Self::Expand(expand) => expand.size(),
            Self::Table(table) => table.size(),
        }
    }
}

#[async_trait]
impl<FD, FS, D, T> SparseAccess<FD, FS, D, T> for SparseAccessor<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        self
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled(txn).await,
            Self::Cast(cast) => cast.filled(txn).await,
            Self::Combine(combine) => combine.filled(txn).await,
            Self::Expand(expand) => expand.filled(txn).await,
            Self::Table(table) => table.filled(txn).await,
        }
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        match self {
            Self::Broadcast(broadcast) => broadcast.filled_count(txn).await,
            Self::Cast(cast) => cast.filled_count(txn).await,
            Self::Combine(combine) => combine.filled_count(txn).await,
            Self::Expand(expand) => expand.filled_count(txn).await,
            Self::Table(table) => table.filled_count(txn).await,
        }
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        match self {
            Self::Broadcast(broadcast) => broadcast.write_value(txn_id, coord, value).await,
            Self::Cast(cast) => cast.write_value(txn_id, coord, value).await,
            Self::Combine(combine) => combine.write_value(txn_id, coord, value).await,
            Self::Expand(expand) => expand.write_value(txn_id, coord, value).await,
            Self::Table(table) => table.write_value(txn_id, coord, value).await,
        }
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for SparseAccessor<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        match self {
            Self::Broadcast(broadcast) => broadcast.read_value_at(txn, coord),
            Self::Cast(cast) => cast.read_value_at(txn, coord),
            Self::Combine(combine) => combine.read_value_at(txn, coord),
            Self::Expand(expand) => expand.read_value_at(txn, coord),
            Self::Table(table) => table.read_value_at(txn, coord),
        }
    }
}

#[derive(Clone)]
pub struct SparseBroadcast<FD, FS, D, T, A> {
    source: A,
    rebase: transform::Broadcast,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    async fn broadcast_coords<'a, S: Stream<Item = TCResult<Coord>> + 'a + Send + Unpin + 'a>(
        self,
        txn: T,
        coords: S,
        num_coords: u64,
    ) -> TCResult<SparseStream<'a>> {
        let broadcast = sorted_values::<FD, T, D, _, _>(txn, self, coords, num_coords).await?;
        Ok(Box::pin(broadcast))
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Broadcast(Box::new(SparseBroadcast {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase.clone();
        let num_coords = self.source.clone().filled_count(txn.clone()).await?;
        let filled = self.source.clone().filled(txn.clone()).await?;

        let filled = filled
            .map_ok(move |(coord, _)| {
                stream::iter(rebase.map_coord(coord).affected().map(TCResult::Ok))
            })
            .try_flatten();

        self.broadcast_coords(txn, filled, num_coords).await
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let rebase = self.rebase;
        let filled = self.source.filled(txn).await?;

        filled
            .try_fold(0u64, move |count, (coord, _)| {
                future::ready(Ok(count + rebase.map_bounds(coord.into()).size()))
            })
            .await
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct SparseCast<FD, FS, D, T, A> {
    source: A,
    dtype: NumberType,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseCast<FD, FS, D, T, A> {
    pub fn new(source: A, dtype: NumberType) -> Self {
        Self {
            source,
            dtype,
            phantom: Phantom::default(),
        }
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseCast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseCast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Cast(Box::new(SparseCast::new(
            self.source.accessor(),
            self.dtype,
        )))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let dtype = self.dtype;

        let filled = self.source.filled(txn).await?;
        let cast = filled.map_ok(move |(coord, value)| (coord, value.into_type(dtype)));
        Ok(Box::pin(cast))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.source.write_value(txn_id, coord, value).await
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseCast<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct SparseCombinator<FD, FS, D, T, L, R> {
    left: L,
    right: R,
    combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, L, R> SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    pub fn new(
        left: L,
        right: R,
        combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<Self> {
        if left.shape() != right.shape() {
            return Err(TCError::unsupported(
                "Tried to combine SparseTensors with different shapes",
            ));
        }

        Ok(SparseCombinator {
            left,
            right,
            combinator,
            dtype,
            phantom: Phantom::default(),
        })
    }

    fn filled_inner<'a>(self, left: SparseStream<'a>, right: SparseStream<'a>) -> SparseStream<'a> {
        let combinator = self.combinator;
        let left_zero = self.left.dtype().zero();
        let right_zero = self.right.dtype().zero();

        let combined =
            SparseCombine::new(self.shape(), left, right).try_filter_map(move |(coord, l, r)| {
                let l = l.unwrap_or_else(|| left_zero.clone());
                let r = r.unwrap_or_else(|| right_zero.clone());
                let value = combinator(l, r);
                let row = if value == value.class().zero() {
                    None
                } else {
                    Some((coord, value))
                };

                future::ready(Ok(row))
            });

        Box::pin(combined)
    }
}

impl<FD, FS, D, T, L, R> TensorAccess for SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.left.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.left.shape()
    }

    fn size(&self) -> u64 {
        self.left.size()
    }
}

#[async_trait]
impl<FD, FS, D, T, L, R> SparseAccess<FD, FS, D, T> for SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Combine(Box::new(SparseCombinator {
            left: self.left.accessor(),
            right: self.right.accessor(),
            combinator: self.combinator,
            dtype: self.dtype,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let left = self.left.clone().filled(txn.clone());
        let right = self.right.clone().filled(txn);
        let (left, right) = try_join!(left, right)?;
        Ok(self.filled_inner(left, right))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        let count = self.filled(txn).await?;

        count
            .try_fold(0u64, |count, _| future::ready(Ok(count + 1)))
            .await
    }

    async fn write_value(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }
}

impl<FD, FS, D, T, L, R> ReadValueAt<D> for SparseCombinator<FD, FS, D, T, L, R>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    L: SparseAccess<FD, FS, D, T>,
    R: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let left = self.left.read_value_at(txn.clone(), coord.to_vec());
            let right = self.right.read_value_at(txn, coord);
            let ((coord, left), (_, right)) = try_join!(left, right)?;
            let value = (self.combinator)(left, right);
            Ok((coord, value))
        })
    }
}

#[derive(Clone)]
pub struct SparseExpand<FD, FS, D, T, A> {
    source: A,
    rebase: transform::Expand,
    phantom: Phantom<FD, FS, D, T>,
}

impl<FD, FS, D, T, A> SparseExpand<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    pub fn new(source: A, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(source.shape().clone(), axis)?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<FD, FS, D, T, A> TensorAccess for SparseExpand<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim() + 1
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.shape().size()
    }
}

#[async_trait]
impl<FD, FS, D, T, A> SparseAccess<FD, FS, D, T> for SparseExpand<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    fn accessor(self) -> SparseAccessor<FD, FS, D, T> {
        SparseAccessor::Expand(Box::new(SparseExpand {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        }))
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase;
        let filled = self.source.filled(txn).await?;
        let filled = filled.map_ok(move |(coord, value)| (rebase.map_coord(coord), value));
        Ok(Box::pin(filled))
    }

    async fn filled_count(self, txn: T) -> TCResult<u64> {
        self.source.filled_count(txn).await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value(txn_id, coord, value).await
    }
}

impl<FD, FS, D, T, A> ReadValueAt<D> for SparseExpand<FD, FS, D, T, A>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FD, FS, D, T>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
    }
}
