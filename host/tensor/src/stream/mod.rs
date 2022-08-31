use std::pin::Pin;

use futures::Future;

use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tc_value::Number;

use super::Coord;

pub use sorted::*;

mod sorted;

pub type Read<'a> = Pin<Box<dyn Future<Output = TCResult<(Coord, Number)>> + Send + 'a>>;

/// Trait defining a read operation for a single [`Tensor`] element
pub trait ReadValueAt<D: Dir> {
    /// The transaction context
    type Txn: Transaction<D>;

    /// Read the value of the element at the given [`Coord`].
    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a>;
}
