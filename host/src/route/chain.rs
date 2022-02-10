use std::convert::TryInto;

use log::debug;
use safecast::{CastFrom, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::Transaction;
use tc_value::Number;
use tcgeneric::{Id, Map, PathSegment, TCPath, Tuple};

use crate::chain::{Chain, ChainInstance, ChainType, Subject, SubjectCollection, SubjectMap};
use crate::state::State;

use super::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route, COPY};

impl Route for ChainType {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        None
    }
}

impl Route for SubjectCollection {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Subject::route {}", TCPath::from(path));

        match self {
            Self::BTree(btree) => btree.route(path),
            Self::Table(table) => table.route(path),
            #[cfg(feature = "tensor")]
            Self::Dense(dense) => dense.route(path),
            #[cfg(feature = "tensor")]
            Self::Sparse(sparse) => sparse.route(path),
        }
    }
}

struct MapHandler<'a> {
    map: &'a Map<Subject>,
}

impl<'a> Handler<'a> for MapHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    return Subject::Map(self.map.clone()).into_state(*txn.id()).await;
                }

                let key = Id::try_cast_from(key, |v| TCError::bad_request("invalid Id", v))?;
                let subject = self
                    .map
                    .get(&key)
                    .cloned()
                    .ok_or_else(|| TCError::not_found(format!("chain subject {}", key)))?;

                subject.into_state(*txn.id()).await
            })
        }))
    }
}

impl<'a> From<&'a Map<Subject>> for MapHandler<'a> {
    fn from(map: &'a Map<Subject>) -> Self {
        Self { map }
    }
}

impl Route for Map<Subject> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Map<Subject> route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(MapHandler::from(self)))
        } else if let Some(subject) = self.get(&path[0]) {
            debug!("Map<Subject> found {} at {}", subject, &path[0]);
            subject.route(&path[1..])
        } else {
            None
        }
    }
}

struct SubjectMapHandler {
    collection: SubjectMap,
}

impl<'a> Handler<'a> for SubjectMapHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let txn_id = *txn.id();

                if key.is_none() {
                    return self.collection.into_state(txn_id).await;
                }

                let id =
                    key.try_cast_into(|v| TCError::bad_request("invalid Id for SubjectMap", v))?;

                let subject = self.collection.get(txn_id, &id).await?;
                let subject = subject.ok_or_else(|| TCError::not_found(id))?;
                Ok(State::Collection(subject.into()))
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, state| {
            Box::pin(async move {
                let id =
                    key.try_cast_into(|v| TCError::bad_request("invalid Id for SubjectMap", v))?;

                let collection = state.try_into()?;
                self.collection.put(txn, id, collection).await
            })
        }))
    }
}

impl From<SubjectMap> for SubjectMapHandler {
    fn from(collection: SubjectMap) -> Self {
        Self { collection }
    }
}

impl Route for SubjectMap {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(SubjectMapHandler::from(self.clone())))
        } else {
            unimplemented!()
        }
    }
}

struct TupleHandler<'a> {
    tuple: &'a Tuple<Subject>,
}

impl<'a> Handler<'a> for TupleHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    return Subject::Tuple(self.tuple.clone())
                        .into_state(*txn.id())
                        .await;
                }

                let i =
                    Number::try_cast_from(key, |v| TCError::bad_request("invalid tuple index", v))?;

                let i = usize::cast_from(i);

                let subject = self
                    .tuple
                    .get(i)
                    .cloned()
                    .ok_or_else(|| TCError::not_found(format!("no such index: {}", i)))?;

                subject.into_state(*txn.id()).await
            })
        }))
    }
}

impl<'a> From<&'a Tuple<Subject>> for TupleHandler<'a> {
    fn from(tuple: &'a Tuple<Subject>) -> Self {
        Self { tuple }
    }
}

impl Route for Tuple<Subject> {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(TupleHandler::from(self)));
        } else if let Some(i) = usize::opt_cast_from(path[0].clone()) {
            if i < self.len() {
                return self[i].route(&path[1..]);
            }
        }

        None
    }
}

impl Route for Subject {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Subject {} route {}", self, TCPath::from(path));

        match self {
            Self::Collection(subject) => subject.route(path),
            Self::Dynamic(subject) => subject.route(path),
            Self::Map(map) => map.route(path),
            Self::Tuple(tuple) => tuple.route(path),
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
                    Err(TCError::not_implemented("chain slicing"))
                }
            })
        }))
    }
}

impl<'a> From<&'a Chain> for ChainHandler<'a> {
    fn from(chain: &'a Chain) -> Self {
        Self { chain }
    }
}

#[allow(unused)]
struct CopyHandler<'a> {
    chain: &'a Chain,
}

impl<'a> Handler<'a> for CopyHandler<'a> {
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

impl<'a> From<&'a Chain> for CopyHandler<'a> {
    fn from(chain: &'a Chain) -> Self {
        Self { chain }
    }
}

impl Route for Chain {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("Chain::route {}", TCPath::from(path));

        if path.len() == 1 && path[0].as_str() == "chain" {
            Some(Box::new(ChainHandler::from(self)))
        } else if path == &COPY[..] {
            Some(Box::new(CopyHandler::from(self)))
        } else {
            Some(Box::new(AppendHandler::new(self, path)))
        }
    }
}
