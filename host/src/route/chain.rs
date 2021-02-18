use generic::PathSegment;
use transact::Transaction;

use crate::chain::{Chain, ChainInstance, Subject};

use super::{GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::route::Public;

impl Route for Chain {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        self.subject().route(path)
    }
}

struct SubjectHandler<'a> {
    subject: &'a Subject,
    path: &'a [PathSegment],
}

impl<'a> SubjectHandler<'a> {
    fn new(subject: &'a Subject, path: &'a [PathSegment]) -> Self {
        Self { subject, path }
    }
}

impl<'a> Handler<'a> for SubjectHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let subject = self.subject.at(txn.id()).await?;
                subject.get(&txn, self.path, key).await
            })
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if self.path.is_empty() {
                    self.subject.put(txn.id(), key, value).await
                } else {
                    let subject = self.subject.at(txn.id()).await?;
                    subject.put(&txn, self.path, key, value).await
                }
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                let subject = self.subject.at(txn.id()).await?;
                subject.post(&txn, self.path, params).await
            })
        }))
    }
}

impl Route for Subject {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(SubjectHandler::new(self, path)))
    }
}
