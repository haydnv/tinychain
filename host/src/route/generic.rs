use std::ops::Deref;
use std::str::FromStr;

use safecast::{CastFrom, TryCastFrom};

use tc_error::TCError;
use tcgeneric::{Id, Instance, Map, PathSegment, Tuple};

use crate::scalar::Number;
use crate::state::State;

use super::{GetHandler, Handler, Route};

struct MapHandler<'a, T: Clone> {
    map: &'a Map<T>,
}

impl<'a, T: Instance + Clone> Handler<'a> for MapHandler<'a, T>
where
    State: From<Map<T>>,
    State: From<T>,
{
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(State::from(self.map.clone()))
                } else {
                    let key = Id::try_cast_from(key, |v| TCError::bad_request("invalid Id", v))?;
                    self.map
                        .get(&key)
                        .cloned()
                        .map(State::from)
                        .ok_or_else(|| TCError::not_found(key))
                }
            })
        }))
    }
}

impl<T: Instance + Route + Clone> Route for Map<T>
where
    State: From<Map<T>>,
    State: From<T>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(MapHandler { map: self }))
        } else if let Some(state) = self.deref().get(&path[0]) {
            state.route(&path[1..])
        } else {
            None
        }
    }
}

struct TupleHandler<'a, T: Clone> {
    tuple: &'a Tuple<T>,
}

impl<'a, T: Instance + Clone> Handler<'a> for TupleHandler<'a, T>
where
    State: From<Tuple<T>>,
    State: From<T>,
{
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(State::from(self.tuple.clone()))
                } else {
                    let i = Number::try_cast_from(key, |v| {
                        TCError::bad_request("invalid tuple index", v)
                    })?;
                    let i = usize::cast_from(i);
                    self.tuple
                        .deref()
                        .get(i)
                        .cloned()
                        .map(State::from)
                        .ok_or_else(|| TCError::not_found(i))
                }
            })
        }))
    }
}

impl<T: Instance + Route + Clone> Route for Tuple<T>
where
    State: From<Tuple<T>>,
    State: From<T>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(TupleHandler { tuple: self }))
        } else if let Ok(i) = usize::from_str(path[0].as_str()) {
            if let Some(state) = self.deref().get(i) {
                state.route(&path[1..])
            } else {
                None
            }
        } else {
            None
        }
    }
}
