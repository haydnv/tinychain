use std::fmt;

use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::{Persist, Restore};
use tc_transact::Transaction;
use tcgeneric::{PathSegment, TCPath};

use crate::chain::{Chain, ChainInstance, ChainType};
use crate::fs;
use crate::state::State;
use crate::txn::Txn;

use super::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route, COPY};

impl Route for ChainType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}

struct AppendHandler<'a, T> {
    chain: &'a Chain<T>,
}

impl<'a, T> AppendHandler<'a, T> {
    fn new(chain: &'a Chain<T>) -> Self {
        Self { chain }
    }
}

impl<'a, T> Handler<'a> for AppendHandler<'a, T>
where
    T: Route + Public,
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

struct ChainHandler<'a, T> {
    chain: &'a Chain<T>,
}

impl<'a, T> Handler<'a> for ChainHandler<'a, T>
where
    T: Send + Sync,
    Chain<T>: Clone,
    State: From<Chain<T>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.chain.clone().into())
                } else {
                    Err(TCError::not_implemented("chain slicing"))
                }
            })
        }))
    }
}

impl<'a, T> From<&'a Chain<T>> for ChainHandler<'a, T> {
    fn from(chain: &'a Chain<T>) -> Self {
        Self { chain }
    }
}

#[allow(unused)]
struct CopyHandler<'a, T> {
    chain: &'a Chain<T>,
}

impl<'a, T> Handler<'a> for CopyHandler<'a, T>
where
    T: Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                key.expect_none()?;
                Err(TCError::not_implemented("copy a Chain"))
            })
        }))
    }
}

impl<'a, T> From<&'a Chain<T>> for CopyHandler<'a, T> {
    fn from(chain: &'a Chain<T>) -> Self {
        Self { chain }
    }
}

impl<T> Route for Chain<T>
where
    T: Persist<fs::Dir, Txn = Txn>
        + Restore<fs::Dir>
        + TryCastFrom<State>
        + Route
        + Public
        + fmt::Display
        + Clone,
    State: From<Chain<T>>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(AppendHandler::new(self)))
        } else if path.len() == 1 && path[0].as_str() == "chain" {
            Some(Box::new(ChainHandler::from(self)))
        } else if path == &COPY[..] {
            Some(Box::new(CopyHandler::from(self)))
        } else {
            self.subject().route(path)
        }
    }
}
