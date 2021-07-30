use async_trait::async_trait;
use destream::en;
use futures::future::{self, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};
use safecast::{CastInto, TryCastFrom, TryCastInto};

use tc_btree::BTreeInstance;
use tc_error::*;
use tc_table::TableStream;
use tc_transact::{IntoView, Transaction};
use tcgeneric::{TCBoxTryFuture, TCBoxTryStream};

use crate::closure::Closure;
use crate::collection::Collection;
use crate::fs;
use crate::state::{State, StateView};
use crate::stream::group::GroupStream;
use crate::txn::Txn;
use crate::value::Value;

mod group;

#[derive(Clone)]
pub enum TCStream {
    Aggregate(Box<TCStream>),
    Collection(Collection),
}

impl TCStream {
    pub fn aggregate(self) -> Self {
        Self::Aggregate(Box::new(self))
    }

    pub async fn fold(self, txn: Txn, mut value: Value, op: Closure) -> TCResult<Value> {
        let mut source = self.into_stream(txn.clone()).await?;
        loop {
            if let Some(state) = source.try_next().await? {
                let mut old_value = Value::None;
                std::mem::swap(&mut old_value, &mut value);
                let state = op
                    .clone()
                    .call(&txn, (old_value, state).cast_into())
                    .await?;

                value = state.try_cast_into(|s| {
                    TCError::bad_request(
                        "closure provided to Stream::fold must return a Value, not",
                        s,
                    )
                })?;
            } else {
                break Ok(value);
            }
        }
    }

    pub async fn for_each(self, txn: &Txn, op: Closure) -> TCResult<()> {
        let stream = self.into_stream(txn.clone()).await?;

        stream
            .map_ok(move |args| op.clone().call(&txn, args))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), _none| future::ready(Ok(())))
            .await
    }

    pub fn into_stream<'a>(self, txn: Txn) -> TCBoxTryFuture<'a, TCBoxTryStream<'static, State>> {
        Box::pin(async move {
            match self {
                Self::Aggregate(source) => {
                    source
                        .into_stream(txn)
                        .map_ok(Self::execute_aggregate)
                        .await
                }
                Self::Collection(collection) => Self::execute_stream(collection, txn).await,
            }
        })
    }

    fn execute_aggregate(source: TCBoxTryStream<'static, State>) -> TCBoxTryStream<'static, State> {
        let values = source.map(|r| {
            r.and_then(|state| {
                Value::try_cast_from(state, |s| {
                    TCError::bad_request("aggregate Stream requires a Value, not {}", s)
                })
            })
        });

        let aggregate: TCBoxTryStream<'static, State> =
            Box::pin(GroupStream::from(values).map_ok(State::from));

        aggregate
    }

    async fn execute_stream(
        collection: Collection,
        txn: Txn,
    ) -> TCResult<TCBoxTryStream<'static, State>> {
        match collection {
            Collection::BTree(btree) => {
                let keys = btree.keys(*txn.id()).await?;
                let keys: TCBoxTryStream<'static, State> =
                    Box::pin(keys.map_ok(Value::from).map_ok(State::from));
                Ok(keys)
            }
            Collection::Table(table) => {
                let rows = table.rows(*txn.id()).await?;
                let rows: TCBoxTryStream<'static, State> =
                    Box::pin(rows.map_ok(Value::from).map_ok(State::from));
                Ok(rows)
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
