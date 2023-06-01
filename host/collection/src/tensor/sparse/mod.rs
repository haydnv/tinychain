mod base;

use async_trait::async_trait;
use futures::Stream;

use tc_error::TCResult;
use tc_transact::TxnId;
use tc_value::Number;

use super::Coord;

pub use base::SparseTensorTable;
pub use fensor::sparse::{IndexSchema, Node, Schema};

#[async_trait]
pub trait SparseTensorRead {
    type Elements: Stream<Item = TCResult<(Coord, Number)>>;

    async fn into_elements(self, txn_id: TxnId) -> TCResult<Self::Elements>;
}
