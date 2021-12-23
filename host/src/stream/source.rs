use async_trait::async_trait;
use futures::stream::TryStreamExt;
use safecast::CastInto;

use tc_btree::BTreeInstance;
use tc_error::*;
use tc_table::TableStream;
use tc_transact::Transaction;
use tc_value::{Number, Value};
use tcgeneric::TCBoxTryStream;

use crate::state::State;
use crate::txn::Txn;

use super::TCStream;

#[async_trait]
pub trait Source: Sized
where
    TCStream: From<Self>,
{
    async fn into_stream(self, txn: Txn) -> TCResult<TCBoxTryStream<'static, State>>;
}

#[derive(Clone)]
pub struct Collection {
    collection: crate::collection::Collection,
}

#[async_trait]
impl Source for Collection {
    async fn into_stream(self, txn: Txn) -> TCResult<TCBoxTryStream<'static, State>> {
        use crate::collection::Collection::*;

        match self.collection {
            BTree(btree) => {
                let keys = btree.keys(*txn.id()).await?;
                let keys: TCBoxTryStream<'static, State> =
                    Box::pin(keys.map_ok(Value::from).map_ok(State::from));

                Ok(keys)
            }
            Table(table) => {
                let rows = table.rows(*txn.id()).await?;
                let rows: TCBoxTryStream<'static, State> =
                    Box::pin(rows.map_ok(Value::from).map_ok(State::from));

                Ok(rows)
            }

            #[cfg(feature = "tensor")]
            Tensor(tensor) => match tensor {
                tc_tensor::Tensor::Dense(dense) => {
                    use tc_tensor::DenseAccess;
                    let elements = dense.into_inner().value_stream(txn).await?;
                    Ok(Box::pin(elements.map_ok(State::from)))
                }
                tc_tensor::Tensor::Sparse(sparse) => {
                    use tc_tensor::SparseAccess;
                    use tcgeneric::Tuple;

                    let filled = sparse.into_inner().filled(txn).await?;
                    let filled = filled
                        .map_ok(|(coord, value)| {
                            let coord = coord
                                .into_iter()
                                .map(Number::from)
                                .collect::<Tuple<Value>>();

                            Tuple::<Value>::from(vec![Value::Tuple(coord), value.cast_into()])
                        })
                        .map_ok(State::from);

                    Ok(Box::pin(filled))
                }
            }
        }
    }
}

impl From<crate::collection::Collection> for Collection {
    fn from(collection: crate::collection::Collection) -> Self {
        Self { collection }
    }
}

impl From<Collection> for TCStream {
    fn from(collection: Collection) -> Self {
        TCStream::Collection(collection)
    }
}
