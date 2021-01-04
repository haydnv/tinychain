use std::pin::Pin;

use futures::Future;

use crate::scalar::Number;
use crate::transaction::Txn;
use crate::TCResult;

use super::Coord;

mod reader;
mod sorted;

pub use reader::*;
pub use sorted::*;

pub type Read<'a> = Pin<Box<dyn Future<Output = TCResult<(Coord, Number)>> + Send + 'a>>;

pub trait ReadValueAt {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a>;
}
