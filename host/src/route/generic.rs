use std::fmt;
use std::iter::Cloned;
use std::ops::Deref;
use std::str::FromStr;

use futures::stream::{self, StreamExt, TryStreamExt};
use safecast::*;

use tc_error::*;
use tc_value::{Number, Value};
use tcgeneric::{label, Id, Instance, Map, PathSegment, Tuple};

use crate::closure::Closure;
use crate::scalar::Scalar;
use crate::state::State;

use super::{GetHandler, Handler, PostHandler, Route};

struct AppendHandler<'a, T: Clone> {
    tuple: &'a Tuple<T>,
}

impl<'a, T> Handler<'a> for AppendHandler<'a, T>
where
    T: fmt::Display + Clone + Send + Sync,
    State: From<T>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if self.tuple.is_empty() {
                    let key = Tuple::<Value>::try_cast_from(key, |v| {
                        TCError::bad_request("not a Tuple", v)
                    })?;

                    return Ok(Value::Tuple(key).into());
                }

                let suffix =
                    Tuple::<Value>::try_cast_from(key, |v| TCError::bad_request("not a Tuple", v))?;

                let items = self.tuple.iter().cloned().map(State::from);
                let items = items.chain(suffix.into_iter().map(Scalar::Value).map(State::Scalar));

                Ok(State::Tuple(items.collect()))
            })
        }))
    }
}

impl<'a, T: Clone> From<&'a Tuple<T>> for AppendHandler<'a, T> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        Self { tuple }
    }
}

struct MapHandler<'a, T: Clone> {
    map: &'a Map<T>,
}

impl<'a, T: Instance + Clone> Handler<'a> for MapHandler<'a, T>
where
    State: From<Map<T>>,
    State: From<T>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
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

struct MapOpHandler<I> {
    len: usize,
    items: I,
}

impl<'a, I> Handler<'a> for MapOpHandler<I>
where
    I: IntoIterator + Send + 'a,
    I::IntoIter: Send + 'a,
    State: From<I::Item>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let op: Closure = params.require(&label("op").into())?;

                let mut tuple = Vec::with_capacity(self.len);
                let mut mapped = stream::iter(self.items)
                    .map(State::from)
                    .map(|state| op.clone().call(txn, state))
                    .buffered(num_cpus::get());

                while let Some(item) = mapped.try_next().await? {
                    tuple.push(item);
                }

                Ok(State::Tuple(tuple.into()))
            })
        }))
    }
}

impl<'a, T: Clone + 'a> From<&'a Tuple<T>> for MapOpHandler<Cloned<std::slice::Iter<'a, T>>> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        let len = tuple.len();
        let items = tuple.iter().cloned();
        Self { len, items }
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
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
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
                        .get(i)
                        .cloned()
                        .map(State::from)
                        .ok_or_else(|| TCError::not_found(i))
                }
            })
        }))
    }
}

struct ZipHandler<'a, T: Clone> {
    keys: &'a Tuple<T>,
}

impl<'a, T> Handler<'a> for ZipHandler<'a, T>
where
    T: Clone + Send + Sync,
    State: From<T>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let values: Tuple<Value> =
                    key.try_cast_into(|v| TCError::bad_request("invalid values for Tuple/zip", v))?;

                if self.keys.len() != values.len() {
                    return Err(TCError::unsupported(format!(
                        "cannot zip {} keys with {} values",
                        self.keys.len(),
                        values.len()
                    )));
                }

                let zipped =
                    self.keys
                        .iter()
                        .cloned()
                        .zip(values.into_iter())
                        .map(|(key, value)| {
                            let key = State::from(key);
                            let value = State::Scalar(Scalar::Value(value));
                            State::Tuple(vec![key, value].into())
                        });

                Ok(State::Tuple(zipped.collect()))
            })
        }))
    }
}

impl<'a, T: Clone> From<&'a Tuple<T>> for ZipHandler<'a, T> {
    fn from(keys: &'a Tuple<T>) -> Self {
        ZipHandler { keys }
    }
}

impl<T> Route for Tuple<T>
where
    T: Instance + Route + Clone + fmt::Display,
    State: From<Tuple<T>>,
    State: From<T>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(TupleHandler { tuple: self }))
        } else if let Ok(i) = usize::from_str(path[0].as_str()) {
            if let Some(state) = self.get(i) {
                state.route(&path[1..])
            } else {
                None
            }
        } else if path.len() == 1 {
            match path[0].as_str() {
                "append" => Some(Box::new(AppendHandler::from(self))),
                "map" => Some(Box::new(MapOpHandler::from(self))),
                "zip" => Some(Box::new(ZipHandler::from(self))),
                _ => None,
            }
        } else {
            None
        }
    }
}
