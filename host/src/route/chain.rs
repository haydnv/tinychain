use log::debug;

use tc_error::*;
use tc_transact::Transaction;
use tcgeneric::{PathSegment, TCPath};

use crate::chain::{Chain, ChainInstance, ChainType, Subject};

use super::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};

impl Route for ChainType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}

impl Route for Subject {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Subject::route {}", TCPath::from(path));

        match self {
            Self::BTree(btree) => btree.route(path),
            Self::Map(map) => map.route(path),
            Self::Table(table) => table.route(path),
            Self::Tuple(tuple) => tuple.route(path),

            #[cfg(feature = "tensor")]
            Self::Dense(dense) => dense.route(path),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => sparse.route(path),
        }
    }
}

struct AppendHandler<'a> {
    chain: &'a Chain,
    path: &'a [PathSegment],
}

impl<'a> AppendHandler<'a> {
    fn new(chain: &'a Chain, path: &'a [PathSegment]) -> Self {
        Self { chain, path }
    }
}

impl<'a> Handler<'a> for AppendHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        match self.chain.subject().route(self.path) {
            Some(handler) => handler.get(),
            None => None,
        }
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        match self.chain.subject().route(self.path) {
            Some(handler) => match handler.put() {
                Some(put_handler) => Some(Box::new(|txn, key, value| {
                    Box::pin(async move {
                        debug!("Chain::put {} <- {}", key, value);

                        let path = self.path.to_vec().into();
                        self.chain
                            .append_put(txn, path, key.clone(), value.clone())
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
        match self.chain.subject().route(self.path) {
            Some(handler) => handler.post(),
            None => None,
        }
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        match self.chain.subject().route(self.path) {
            Some(handler) => match handler.delete() {
                Some(delete_handler) => Some(Box::new(|txn, key| {
                    Box::pin(async move {
                        debug!("Chain::delete {}", key);

                        self.chain
                            .append_delete(*txn.id(), self.path.to_vec().into(), key.clone())
                            .await?;

                        delete_handler(txn, key).await
                    })
                })),
                None => None,
            },
            None => None,
        }
    }
}

struct ChainHandler<'a> {
    chain: &'a Chain,
}

impl<'a> From<&'a Chain> for ChainHandler<'a> {
    fn from(chain: &'a Chain) -> Self {
        Self { chain }
    }
}

impl<'a> Handler<'a> for ChainHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(self.chain.clone().into())
                } else {
                    Err(TCError::bad_request("invalid key for Chain", key))
                }
            })
        }))
    }
}

impl Route for Chain {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.len() == 1 && path[0].as_str() == "chain" {
            Some(Box::new(ChainHandler::from(self)))
        } else {
            Some(Box::new(AppendHandler::new(self, path)))
        }
    }
}
