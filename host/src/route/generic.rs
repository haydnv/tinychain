use std::convert::TryInto;
use std::fmt;
use std::iter::Cloned;
use std::ops::Deref;
use std::str::FromStr;

use futures::stream::{self, FuturesOrdered, StreamExt, TryStreamExt};
use log::debug;
use safecast::*;

use tc_error::*;
use tc_value::{Number, Value};
use tcgeneric::{label, Id, Instance, Map, PathSegment, TCPath, TCPathBuf, Tuple};

use crate::closure::Closure;
use crate::route::COPY;
use crate::scalar::Scalar;
use crate::state::State;

use super::{AttributeHandler, GetHandler, Handler, PostHandler, Public, Route};

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

struct EqMapHandler<T> {
    map: Map<T>,
}

impl<'a, T: fmt::Display + Send + Sync + 'a> Handler<'a> for EqMapHandler<T>
where
    State: From<T>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let other: Map<State> = params.require(&label("eq").into())?;
                params.expect_empty()?;

                if self.map.len() != other.len() {
                    return Ok(false.into());
                }

                const ERR: &str = "cannot cast into Value from";
                let eq = self
                    .map
                    .into_iter()
                    .zip(other)
                    .map(|((this_id, this), (that_id, that))| {
                        if this_id != that_id {
                            return Ok(false);
                        }

                        let this = State::from(this);
                        let this = Value::try_cast_from(this, |s| TCError::bad_request(ERR, s))?;
                        let that = Value::try_cast_from(that, |s| TCError::bad_request(ERR, s))?;
                        Ok(this == that)
                    })
                    .collect::<TCResult<Vec<bool>>>()?;

                Ok(eq.into_iter().all(|eq| eq).into())
            })
        }))
    }
}

impl<T> From<Map<T>> for EqMapHandler<T> {
    fn from(map: Map<T>) -> Self {
        Self { map }
    }
}

struct EqTupleHandler<T> {
    tuple: Tuple<T>,
}

impl<'a, T: fmt::Display + Send + Sync + 'a> Handler<'a> for EqTupleHandler<T>
where
    State: From<T>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let other: Tuple<State> = params.require(&label("eq").into())?;
                params.expect_empty()?;

                if self.tuple.len() != other.len() {
                    return Ok(false.into());
                }

                const ERR: &str = "cannot cast into Value from";
                let eq = self
                    .tuple
                    .into_iter()
                    .zip(other)
                    .map(|(this, that)| {
                        let this = State::from(this);
                        let this = Value::try_cast_from(this, |s| TCError::bad_request(ERR, s))?;
                        let that = Value::try_cast_from(that, |s| TCError::bad_request(ERR, s))?;
                        Ok(this == that)
                    })
                    .collect::<TCResult<Vec<bool>>>()?;

                Ok(eq.into_iter().all(|eq| eq).into())
            })
        }))
    }
}

impl<T> From<Tuple<T>> for EqTupleHandler<T> {
    fn from(tuple: Tuple<T>) -> Self {
        Self { tuple }
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
                    self.map.get(&key).cloned().map(State::from).ok_or_else(|| {
                        let msg = format!(
                            "{} in Map with keys {}",
                            key,
                            self.map.keys().collect::<Tuple<&Id>>()
                        );

                        TCError::not_found(msg)
                    })
                }
            })
        }))
    }
}

impl<T: Instance + Route + Clone + fmt::Display> Route for Map<T>
where
    State: From<Map<T>>,
    State: From<T>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        debug!("{} route {}", self, TCPath::from(path));

        if !path.is_empty() {
            debug!(
                "{} contains {}? {}",
                self,
                path[0],
                self.contains_key(&path[0])
            );
        }

        if path.is_empty() {
            Some(Box::new(MapHandler { map: self }))
        } else if let Some(state) = self.deref().get(&path[0]) {
            debug!("member {} route {}", state, TCPath::from(&path[1..]));
            state.route(&path[1..])
        } else if path.len() == 1 {
            match path[0].as_str() {
                "eq" => Some(Box::new(EqMapHandler::from(self.clone()))),
                "len" => Some(Box::new(AttributeHandler::from(Number::from(
                    self.len() as u64
                )))),
                _ => None,
            }
        } else {
            None
        }
    }
}

struct TupleCopyHandler<'a, T> {
    tuple: &'a Tuple<T>,
}

impl<'a, T> Handler<'a> for TupleCopyHandler<'a, T>
where
    T: Route + Clone + Send + fmt::Display + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                let path = TCPathBuf::from(COPY);
                let mut copies: FuturesOrdered<_> = self
                    .tuple
                    .iter()
                    .map(|item| Public::get(item, txn, &path, Value::default()))
                    .collect();

                let mut copy = Vec::with_capacity(self.tuple.len());
                while let Some(item) = copies.try_next().await? {
                    copy.push(item);
                }

                Ok(State::Tuple(copy.into()))
            })
        }))
    }
}

impl<'a, T> From<&'a Tuple<T>> for TupleCopyHandler<'a, T> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        Self { tuple }
    }
}

struct TupleFoldHandler<'a, T> {
    tuple: &'a Tuple<T>,
}

impl<'a, T> Handler<'a> for TupleFoldHandler<'a, T>
where
    T: Clone + Send + Sync + 'a,
    State: From<T>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let item_name: Id = params.require(&label("item_name").into())?;
                let op: Closure = params.require(&label("op").into())?;
                let mut state: State = params.require(&label("value").into())?;
                params.expect_empty()?;

                for item in self.tuple.iter().cloned() {
                    let mut params: Map<State> = state.try_into()?;
                    params.insert(item_name.clone(), item.into());
                    state = op.clone().call(txn, params.into()).await?;
                }

                Ok(state.into())
            })
        }))
    }
}

impl<'a, T: Clone + 'a> From<&'a Tuple<T>> for TupleFoldHandler<'a, T> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        Self { tuple }
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
                        .ok_or_else(|| TCError::not_found(format!("no such index: {}", i)))
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
            if let Some(state) = self.deref().get(i) {
                state.route(&path[1..])
            } else {
                None
            }
        } else if path.len() == 1 {
            match path[0].as_str() {
                "append" => Some(Box::new(AppendHandler::from(self))),
                "copy" => Some(Box::new(TupleCopyHandler::from(self))),
                "fold" => Some(Box::new(TupleFoldHandler::from(self))),
                "len" => Some(Box::new(AttributeHandler::from(Number::from(
                    self.len() as u64
                )))),
                "eq" => Some(Box::new(EqTupleHandler::from(self.clone()))),
                "map" => Some(Box::new(MapOpHandler::from(self))),
                "zip" => Some(Box::new(ZipHandler::from(self))),
                _ => None,
            }
        } else {
            None
        }
    }
}
