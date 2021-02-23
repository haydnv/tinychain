use log::debug;

use tc_transact::Transaction;
use tcgeneric::{PathSegment, TCPath};

use crate::chain::{Chain, ChainInstance, Subject};

use super::{GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::route::Public;

impl Route for Chain {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        let subject = self.subject();
        if path.is_empty() {
            // TODO: chain methods
            subject.route(path)
        } else if path[0].as_str() == "subject" {
            subject.route(&path[1..])
        } else {
            // TODO: chain methods
            subject.route(path)
        }
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
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!("Subject::get {} {}", TCPath::from(self.path), key);
                let subject = self.subject.at(txn.id()).await?;
                debug!("Subject is {}", subject);
                subject.get(&txn, self.path, key).await
            })
        }))
    }

    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("Subject::put {} <- {}", key, value);
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
                debug!("Subject::post {}", params);
                let subject = self.subject.at(txn.id()).await?;
                subject.post(&txn, self.path, params).await
            })
        }))
    }
}

impl Route for Subject {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Subject::route {}", TCPath::from(path));
        Some(Box::new(SubjectHandler::new(self, path)))
    }
}
