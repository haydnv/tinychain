use std::pin::Pin;

use futures::Future;

use crate::general::TCResult;
use crate::scalar::Number;
use crate::transaction::Txn;

mod reader;
mod sorted;

pub use reader::*;
pub use sorted::*;

pub type Read<'a> = Pin<Box<dyn Future<Output = TCResult<(Vec<u64>, Number)>> + Send + 'a>>;

pub trait ReadValueAt {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Vec<u64>) -> Read<'a>;
}
