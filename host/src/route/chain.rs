use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::fs::File;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::{Instance, PathSegment, TCPath};

use crate::chain::{Chain, ChainInstance, ChainType, Subject, SUBJECT};

use super::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route};

impl Route for ChainType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}

impl Route for Subject {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));
        Some(Box::new(SubjectHandler::new(self, path)))
    }
}

struct SubjectHandler<'a> {
    subject: &'a Subject,
    path: &'a [PathSegment],
}

impl<'a> SubjectHandler<'a> {
    fn new(subject: &'a Subject, path: &'a [PathSegment]) -> Self {
        debug!("SubjectHandler {}", TCPath::from(path));
        Self { subject, path }
    }
}

impl<'a> Handler<'a> for SubjectHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!("Subject::get {} {}", TCPath::from(self.path), key);
                match self.subject {
                    Subject::BTree(btree) => btree.get(&txn, self.path, key).await,
                    Subject::Table(table) => Public::get(table, &txn, self.path, key).await,
                    #[cfg(feature = "tensor")]
                    Subject::Dense(tensor) => tensor.get(&txn, self.path, key).await,
                    #[cfg(feature = "tensor")]
                    Subject::Sparse(tensor) => tensor.get(&txn, self.path, key).await,
                    Subject::Tuple(tuple) => todo!(),
                    Subject::Value(file) => {
                        let value = file.read_block(*txn.id(), SUBJECT.into()).await?;

                        if self.path.is_empty() {
                            Ok(value.clone().into())
                        } else {
                            value.get(&txn, self.path, key).await
                        }
                    }
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                match self.subject {
                    Subject::BTree(btree) => btree.put(&txn, self.path, key, value).await,
                    Subject::Table(table) => table.put(&txn, self.path, key, value).await,
                    #[cfg(feature = "tensor")]
                    Subject::Dense(tensor) => tensor.put(&txn, self.path, key, value).await,
                    #[cfg(feature = "tensor")]
                    Subject::Sparse(tensor) => tensor.put(&txn, self.path, key, value).await,
                    Subject::Value(file) if self.path.is_empty() => {
                        let mut subject = file.write_block(*txn.id(), SUBJECT.into()).await?;

                        let value = Value::try_cast_from(value, |s| {
                            TCError::bad_request(
                                format!("invalid Value {} for Chain subject, expected", s),
                                subject.class(),
                            )
                        })?;

                        *subject = value;

                        Ok(())
                    }
                    Subject::Tuple(tuple) => todo!(),
                    Subject::Value(file) => {
                        let subject = file.read_block(*txn.id(), SUBJECT.into()).await?;

                        subject.put(&txn, self.path, key, value).await
                    }
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                debug!("Subject::post {}", params);
                match self.subject {
                    Subject::BTree(btree) => btree.post(&txn, self.path, params).await,
                    Subject::Table(table) => table.post(&txn, self.path, params).await,
                    #[cfg(feature = "tensor")]
                    Subject::Dense(tensor) => tensor.post(&txn, self.path, params).await,
                    #[cfg(feature = "tensor")]
                    Subject::Sparse(tensor) => tensor.post(&txn, self.path, params).await,
                    Subject::Tuple(tuple) => todo!(),
                    Subject::Value(file) => {
                        let subject = file.read_block(*txn.id(), SUBJECT.into()).await?;

                        subject.post(&txn, self.path, params).await
                    }
                }
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                match self.subject {
                    Subject::BTree(btree) => btree.delete(&txn, self.path, key).await,
                    Subject::Table(table) => Public::delete(table, &txn, self.path, key).await,
                    #[cfg(feature = "tensor")]
                    Subject::Dense(tensor) => tensor.delete(&txn, self.path, key).await,
                    #[cfg(feature = "tensor")]
                    Subject::Sparse(tensor) => tensor.delete(&txn, self.path, key).await,
                    Subject::Tuple(tuple) => todo!(),
                    Subject::Value(file) if self.path.is_empty() => {
                        let mut subject = file.write_block(*txn.id(), SUBJECT.into()).await?;
                        *subject = Value::None;
                        Ok(())
                    }
                    Subject::Value(file) => {
                        let subject = file.read_block(*txn.id(), SUBJECT.into()).await?;
                        subject.delete(&txn, self.path, key).await
                    }
                }
            })
        }))
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
