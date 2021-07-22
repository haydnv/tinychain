use async_trait::async_trait;
use destream::en;
use futures::future;
use futures::stream::{StreamExt, TryStreamExt};

use tc_btree::BTreeInstance;
use tc_error::*;
use tc_table::TableInstance;
use tc_transact::{IntoView, Transaction};
use tcgeneric::TCBoxTryStream;

use crate::closure::Closure;
use crate::collection::Collection;
use crate::fs;
use crate::scalar::OpDef;
use crate::state::{State, StateView};
use crate::txn::Txn;
use crate::value::Value;

#[derive(Clone)]
pub enum TCStream {
    Collection(Collection),
}

impl TCStream {
    pub async fn for_each(self, txn: Txn, op: Closure) -> TCResult<()> {
        let stream = self.into_stream(txn.clone()).await?;

        stream
            .map(move |r| r.and_then(|state| op.clone().into_callable(state)))
            .map_ok(|(context, op_def)| OpDef::call(op_def, &txn, context))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), _none| future::ready(Ok(())))
            .await
    }

    pub async fn into_stream(self, txn: Txn) -> TCResult<TCBoxTryStream<'static, State>> {
        match self {
            Self::Collection(collection) => match collection {
                Collection::BTree(btree) => {
                    let keys = btree.keys(*txn.id()).await?;
                    Ok(Box::pin(keys.map_ok(Value::from).map_ok(State::from)))
                }
                Collection::Table(table) => {
                    let rows = table.rows(*txn.id()).await?;
                    Ok(Box::pin(rows.map_ok(Value::from).map_ok(State::from)))
                }

                #[cfg(feature = "tensor")]
                Collection::Tensor(tensor) => match tensor {
                    tc_tensor::Tensor::Dense(_dense) => {
                        Err(TCError::not_implemented("DenseTensor::into_stream"))
                    }
                    tc_tensor::Tensor::Sparse(_sparse) => {
                        Err(TCError::not_implemented("SparseTensor::into_stream"))
                    }
                },
            },
        }
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for TCStream {
    type Txn = Txn;
    type View = en::SeqStream<TCResult<StateView<'en>>, TCBoxTryStream<'en, StateView<'en>>>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let stream = self.into_stream(txn.clone()).await?;
        let view_stream: TCBoxTryStream<'en, StateView<'en>> = Box::pin(
            stream
                .map_ok(move |state| state.into_view(txn.clone()))
                .try_buffered(num_cpus::get()),
        );

        Ok(en::SeqStream::from(view_stream))
    }
}

impl<T> From<T> for TCStream
where
    Collection: From<T>,
{
    fn from(collection: T) -> Self {
        Self::Collection(collection.into())
    }
}
