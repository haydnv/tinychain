use std::marker::PhantomData;
use std::pin::Pin;

use futures::Stream;

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::Transaction;
use tc_value::{Number, NumberType};

use super::{Coord, Shape, TensorAccess};

pub use access::{SparseAccess, SparseAccessor};
pub use table::SparseTable;

mod access;
mod table;

pub type SparseRow = (Coord, Number);
pub type SparseStream<'a> = Pin<Box<dyn Stream<Item = TCResult<SparseRow>> + Send + Unpin + 'a>>;

#[derive(Clone)]
pub struct SparseTensor<F: File<Node>, D: Dir, T: Transaction<D>, A: SparseAccess<F, D, T>> {
    accessor: A,
    file: PhantomData<F>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F: File<Node>, D: Dir, T: Transaction<D>, A: SparseAccess<F, D, T>> TensorAccess
    for SparseTensor<F, D, T, A>
{
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn ndim(&self) -> usize {
        self.accessor.ndim()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }

    fn size(&self) -> u64 {
        self.accessor.size()
    }
}
