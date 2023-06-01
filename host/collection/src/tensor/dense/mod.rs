use async_trait::async_trait;
use fensor::ha_ndarray;
use futures::stream::Stream;

use tc_error::TCResult;
use tc_transact::TxnId;

mod base;

#[async_trait]
pub trait DenseTensorRead {
    type Block: ha_ndarray::NDArrayRead;
    type Blocks: Stream<Item = TCResult<Self::Block>>;

    async fn into_blocks(self, txn_id: TxnId) -> TCResult<Self::Blocks>;
}
