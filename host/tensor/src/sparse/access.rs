use async_trait::async_trait;

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::{Number, NumberType};

use crate::stream::{Read, ReadValueAt};
use crate::{Coord, Shape, TensorAccess};

use super::{SparseStream, SparseTable};

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
