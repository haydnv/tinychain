use std::fmt;
use std::marker::PhantomData;

use log::debug;
use safecast::{AsType, TryCastFrom};

use tc_collection::btree::Node as BTreeNode;
use tc_collection::tensor::{DenseCacheFile, Node as TensorNode};
use tc_collection::Collection;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::public::*;
use tc_transact::Transaction;
use tcgeneric::{PathSegment, TCPath};

use super::{BlockChain, Chain, ChainBlock, ChainInstance, ChainType};

impl<State: StateInstance> Route<State> for ChainType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        None
    }
}

struct AppendHandler<'a, C, T> {
    chain: &'a C,
    phantom: PhantomData<T>,
}

impl<'a, C, T> AppendHandler<'a, C, T> {
    fn new(chain: &'a C) -> Self {
        Self {
            chain,
            phantom: PhantomData,
        }
    }
}

impl<'a, State, C, T> Handler<'a, State> for AppendHandler<'a, C, T>
where
    State: StateInstance,
    C: ChainInstance<State, T> + Send + Sync + 'a,
    T: Route<State> + fmt::Debug + 'a,
    Chain<State, T>: ChainInstance<State, T>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        let handler = self.chain.subject().route(&[])?;
        handler.get()
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        match self.chain.subject().route(&[]) {
            Some(handler) => match handler.put() {
                Some(put_handler) => Some(Box::new(|txn, key, value| {
                    Box::pin(async move {
                        debug!("Chain::put {} <- {:?}", key, value);

                        self.chain
                            .append_put(txn, key.clone(), value.clone())
                            .await?;

                        put_handler(txn, key, value).await
                    })
                })),
                None => None,
            },
            None => None,
        }
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        let handler = self.chain.subject().route(&[])?;
        handler.post()
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, State::Txn>>
    where
        'b: 'a,
    {
        match self.chain.subject().route(&[]) {
            Some(handler) => match handler.delete() {
                Some(delete_handler) => Some(Box::new(|txn, key| {
                    Box::pin(async move {
                        debug!("Chain::delete {}", key);

                        self.chain.append_delete(*txn.id(), key.clone()).await?;

                        delete_handler(txn, key).await
                    })
                })),
                None => None,
            },
            None => None,
        }
    }
}

impl<State, T> Route<State> for Chain<State, T>
where
    State: StateInstance,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + Clone + fmt::Debug,
    Self: ChainInstance<State, T>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(AppendHandler::new(self)))
        } else {
            self.subject().route(path)
        }
    }
}

impl<State, T> Route<State> for BlockChain<State, T>
where
    State: StateInstance,

    State::FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
    T: fs::Persist<State::FE, Txn = State::Txn> + Route<State> + Clone + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(AppendHandler::new(self)))
        } else {
            self.subject().route(path)
        }
    }
}
