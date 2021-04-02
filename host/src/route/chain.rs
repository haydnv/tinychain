use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::Transaction;
use tcgeneric::{PathSegment, TCPath};

use crate::chain::{Chain, ChainInstance};
use crate::scalar::Scalar;

use super::{GetHandler, Handler, PostHandler, Public, PutHandler, Route};

impl Route for Chain {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.is_empty() {
            // TODO: chain methods
            Some(Box::new(SubjectHandler::new(self, path)))
        } else if path[0].as_str() == "chain" {
            // TODO: chain methods
            None
        } else {
            None
        }
    }
}

struct SubjectHandler<'a> {
    chain: &'a Chain,
    path: &'a [PathSegment],
}

impl<'a> SubjectHandler<'a> {
    fn new(chain: &'a Chain, path: &'a [PathSegment]) -> Self {
        debug!("SubjectHandler {}", TCPath::from(path));
        Self { chain, path }
    }
}

impl<'a> Handler<'a> for SubjectHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!("Subject::get {} {}", TCPath::from(self.path), key);
                let subject = self.chain.subject().at(txn.id()).await?;
                debug!("Subject is {}", subject);
                subject.get(&txn, self.path, key).await
            })
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                let subject = self.chain.subject();

                let scalar_value = Scalar::try_cast_from(value.clone(), |v| {
                    TCError::not_implemented(format!("update Chain with value {}", v))
                })?;

                debug!("Subject::put {} <- {}", key, value);
                self.chain
                    .append(
                        *txn.id(),
                        self.path.to_vec().into(),
                        key.clone(),
                        scalar_value,
                    )
                    .await?;

                if self.path.is_empty() {
                    subject
                        .put(*txn.id(), self.path.to_vec().into(), key, value)
                        .await
                } else {
                    let subject = self.chain.subject().at(txn.id()).await?;
                    subject.put(&txn, self.path, key, value).await
                }
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                debug!("Subject::post {}", params);
                let subject = self.chain.subject().at(txn.id()).await?;
                subject.post(&txn, self.path, params).await
            })
        }))
    }
}
