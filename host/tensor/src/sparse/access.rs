use std::convert::TryFrom;

use afarray::Array;
use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, TryStreamExt};

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::{Number, NumberType};

use crate::stream::{sorted_values, Read, ReadValueAt};
use crate::transform::{self, Rebase};
use crate::{Coord, Shape, TensorAccess, TensorType, ERR_NONBIJECTIVE_WRITE};

use super::{Phantom, SparseStream, SparseTable};

#[async_trait]
pub trait SparseAccess<F: File<Node>, D: Dir, T: Transaction<D>>:
    Clone + ReadValueAt<D, Txn = T> + TensorAccess + Send + Sync + 'static
{
    fn accessor(self) -> SparseAccessor<F, D, T>;

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>>;

    async fn filled_count(&self, txn: &T) -> TCResult<u64>;

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()>;
}

#[derive(Clone)]
pub enum SparseAccessor<F, D, T> {
    Table(SparseTable<F, D, T>),
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> TensorAccess for SparseAccessor<F, D, T> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Table(table) => table.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Table(table) => table.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::Table(table) => table.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Table(table) => table.size(),
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> SparseAccess<F, D, T> for SparseAccessor<F, D, T> {
    fn accessor(self) -> SparseAccessor<F, D, T> {
        self
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        match self {
            Self::Table(table) => table.filled(txn).await,
        }
    }

    async fn filled_count(&self, txn: &T) -> TCResult<u64> {
        match self {
            Self::Table(table) => table.filled_count(txn).await,
        }
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        match self {
            Self::Table(table) => table.write_value(txn_id, coord, value).await,
        }
    }
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> ReadValueAt<D> for SparseAccessor<F, D, T> {
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        match self {
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
    A: SparseAccess<FS, D, T>,
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
    A: SparseAccess<FS, D, T>,
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
impl<FD, FS, D, T, A> SparseAccess<FS, D, T> for SparseBroadcast<FD, FS, D, T, A>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FS, D, T>,
    D::FileClass: From<TensorType>,
{
    fn accessor(self) -> SparseAccessor<FS, D, T> {
        // SparseAccessor::Broadcast(Box::new(SparseBroadcast {
        //     source: self.source.accessor(),
        //     rebase: self.rebase,
        //     phantom: Phantom::default(),
        // }))
        todo!()
    }

    async fn filled<'a>(self, txn: T) -> TCResult<SparseStream<'a>> {
        let rebase = self.rebase.clone();
        let num_coords = self.source.filled_count(&txn).await?;
        let filled = self.source.clone().filled(txn.clone()).await?;

        let filled = filled
            .map_ok(move |(coord, _)| {
                stream::iter(rebase.map_coord(coord).affected().map(TCResult::Ok))
            })
            .try_flatten();

        self.broadcast_coords(txn, filled, num_coords).await
    }

    async fn filled_count(&self, txn: &T) -> TCResult<u64> {
        let rebase = &self.rebase;
        let filled = self.source.clone().filled(txn.clone()).await?;

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
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<FS, D, T>,
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
