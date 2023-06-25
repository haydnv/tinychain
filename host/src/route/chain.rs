use std::fmt;
use std::marker::PhantomData;

use log::debug;

use tc_chain::{ChainInstance, ChainType};
use tc_fs::CacheBlock;
use tc_transact::fs::Persist;
use tc_transact::public::Handler;
use tc_transact::Transaction;
use tcgeneric::{PathSegment, TCPath};

use crate::chain::{BlockChain, Chain};
use crate::cluster::Replica;
use crate::state::State;
use crate::txn::Txn;

use super::{DeleteHandler, GetHandler, PostHandler, PutHandler, Route};

impl Route<State> for ChainType {
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

impl<'a, C, T> Handler<'a, State> for AppendHandler<'a, C, T>
where
    C: ChainInstance<State, T> + Send + Sync + 'a,
    T: Route<State> + fmt::Debug + 'a,
    Chain<T>: ChainInstance<State, T>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        let handler = self.chain.subject().route(&[])?;
        handler.get()
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        match self.chain.subject().route(&[]) {
            Some(handler) => match handler.put() {
                Some(put_handler) => Some(Box::new(|txn, key, value| {
                    Box::pin(async move {
                        debug!("Chain::put {} <- {:?}", key, value);

                        self.chain
                            .append_put(*txn.id(), key.clone(), value.clone())
                            .await?;

                        put_handler(txn, key, value).await
                    })
                })),
                None => None,
            },
            None => None,
        }
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        let handler = self.chain.subject().route(&[])?;
        handler.post()
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
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

struct ChainHandler<'a, C, T> {
    chain: &'a C,
    phantom: PhantomData<T>,
}

impl<'a, C, T> ChainHandler<'a, C, T> {
    fn new(chain: &'a C) -> Self {
        Self {
            chain,
            phantom: PhantomData,
        }
    }
}

impl<'a, C, T> Handler<'a, State> for ChainHandler<'a, C, T>
where
    T: Send + Sync + 'a,
    C: ChainInstance<State, T> + Replica + Clone + Send + Sync + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;
                self.chain.state(*txn.id()).await
            })
        }))
    }
}

impl<T> Route<State> for Chain<T>
where
    T: Persist<CacheBlock, Txn = Txn> + Route<State> + Clone + fmt::Debug,
    Self: ChainInstance<State, T> + Replica,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(AppendHandler::new(self)))
        } else if path.len() == 1 && path[0].as_str() == "chain" {
            Some(Box::new(ChainHandler::new(self)))
        } else {
            self.subject().route(path)
        }
    }
}

impl<T> Route<State> for BlockChain<T>
where
    T: Persist<CacheBlock, Txn = Txn> + Route<State> + Clone + fmt::Debug,
    Self: ChainInstance<State, T> + Replica,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(AppendHandler::new(self)))
        } else if path.len() == 1 && path[0].as_str() == "chain" {
            Some(Box::new(ChainHandler::new(self)))
        } else {
            self.subject().route(path)
        }
    }
}
