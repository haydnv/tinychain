use async_trait::async_trait;
use destream::de::Error;
use futures::future::{self, TryFutureExt};
use futures::stream::{StreamExt, TryStreamExt};
use safecast::TryCastFrom;

use tc_btree::BTreeInstance;
use tc_error::*;
use tc_table::TableStream;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::TCBoxTryStream;

use crate::closure::Closure;
use crate::state::State;
use crate::txn::Txn;

use super::TCStream;

/// Trait defining the source of a [`TCStream`].
#[async_trait]
pub trait Source: Clone + Sized
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
                    use safecast::CastInto;
                    use tc_tensor::SparseAccess;
                    use tcgeneric::Tuple;

                    let filled = sparse.into_inner().filled(txn).await?;
                    let filled = filled
                        .map_ok(|(coord, value)| {
                            let coord = coord
                                .into_iter()
                                .map(tc_value::Number::from)
                                .collect::<Tuple<Value>>();

                            Tuple::<Value>::from(vec![Value::Tuple(coord), value.cast_into()])
                        })
                        .map_ok(State::from);

                    Ok(Box::pin(filled))
                }
            },
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

#[derive(Clone)]
pub struct Filter {
    source: TCStream,
    op: Closure,
}

impl Filter {
    pub fn new(source: TCStream, op: Closure) -> Self {
        Self { source, op }
    }
}

#[async_trait]
impl Source for Filter {
    async fn into_stream(self, txn: Txn) -> TCResult<TCBoxTryStream<'static, State>> {
        let source = self.source.into_stream(txn.clone()).await?;
        let op = self.op;

        let filtered = source
            .map_ok(move |state| {
                op.clone()
                    .call_owned(txn.clone(), state.clone())
                    .map_ok(|filter| (filter, state))
            })
            .try_buffered(num_cpus::get())
            .map(|result| {
                result.and_then(|(filter, state)| {
                    bool::try_cast_from(filter, |s| {
                        TCError::invalid_type(s, "a boolean Stream filter condition")
                    })
                    .map(|filter| (filter, state))
                })
            })
            .try_filter_map(|(filter, state)| {
                future::ready({
                    if filter {
                        Ok(Some(state))
                    } else {
                        Ok(None)
                    }
                })
            });

        Ok(Box::pin(filtered))
    }
}

impl From<Filter> for TCStream {
    fn from(filter: Filter) -> Self {
        Self::Filter(Box::new(filter))
    }
}

#[derive(Clone)]
pub struct Flatten {
    source: TCStream,
}

impl Flatten {
    pub fn new(source: TCStream) -> Self {
        Self { source }
    }
}

#[async_trait]
impl Source for Flatten {
    async fn into_stream(self, txn: Txn) -> TCResult<TCBoxTryStream<'static, State>> {
        let source = self.source.into_stream(txn.clone()).await?;

        let flat = source
            .map(|result| {
                result.and_then(|state| {
                    TCStream::try_cast_from(state, |s| {
                        TCError::invalid_type(s, "a Stream of Streams to flatten")
                    })
                })
            })
            .map_ok(move |stream| stream.into_stream(txn.clone()))
            .try_buffered(num_cpus::get())
            .try_flatten();

        Ok(Box::pin(flat))
    }
}

impl From<Flatten> for TCStream {
    fn from(flatten: Flatten) -> TCStream {
        TCStream::Flatten(Box::new(flatten))
    }
}
