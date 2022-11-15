use std::fmt;
use std::marker::PhantomData;

use log::debug;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::Transaction;
use tcgeneric::{PathSegment, TCPath};

use crate::chain::{BlockChain, Chain, ChainInstance, ChainType};
use crate::fs;
use crate::txn::Txn;

use super::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route};

impl Route for ChainType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
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

impl<'a, C, T> Handler<'a> for AppendHandler<'a, C, T>
where
    C: ChainInstance<T> + Send + Sync + 'a,
    T: Route + Public + 'a,
    Chain<T>: ChainInstance<T>,
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
                        debug!("Chain::put {} <- {}", key, value);

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

impl<'a, C, T> Handler<'a> for ChainHandler<'a, C, T>
where
    C: ChainInstance<T> + Clone + Send + Sync + 'a,
    T: Send + Sync + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, _key| {
            Box::pin(async move { Err(TCError::not_implemented("replicate a Chain")) })
        }))
    }
}

impl<T> Route for Chain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Route + Public + fmt::Display + Clone,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
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

impl<T> Route for BlockChain<T>
where
    T: Persist<fs::Dir, Txn = Txn> + Route + Public + fmt::Display + Clone,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
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
