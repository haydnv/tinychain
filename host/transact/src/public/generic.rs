use std::convert::TryInto;
use std::fmt;
use std::iter::Cloned;
use std::marker::PhantomData;
use std::ops::{Bound, Deref};
use std::str::FromStr;

use futures::future::TryFutureExt;
use futures::stream::{FuturesOrdered, TryStreamExt};
use safecast::*;

use tc_error::*;
use tc_value::{Number, UInt, Value};
use tcgeneric::{
    label, path_label, Id, Instance, Map, PathLabel, PathSegment, TCPathBuf, ThreadSafe, Tuple,
};

use crate::Transaction;

use super::helpers::AttributeHandler;
use super::{GetHandler, Handler, PostHandler, Public, Route, StateInstance};

const COPY: PathLabel = path_label(&["copy"]);

struct AppendHandler<'a, State, T: Clone> {
    tuple: &'a Tuple<T>,
    state: PhantomData<State>,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for AppendHandler<'a, State, T>
where
    T: fmt::Debug + Clone + Send + Sync,
    State: From<T> + StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if self.tuple.is_empty() {
                    let key =
                        Tuple::<Value>::try_cast_from(key, |v| TCError::unexpected(v, "a Tuple"))?;

                    return Ok(Value::Tuple(key).into());
                }

                let suffix =
                    Tuple::<Value>::try_cast_from(key, |v| TCError::unexpected(v, "a Tuple"))?;

                let items = self.tuple.iter().cloned().map(State::from);
                let items = items.chain(suffix.into_iter().map(State::from));

                Ok(State::from(items.collect::<Tuple<State>>()))
            })
        }))
    }
}

struct ContainsHandler<'a, State, T> {
    tuple: &'a Tuple<T>,
    state: PhantomData<State>,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for ContainsHandler<'a, State, T>
where
    T: Clone + Send + Sync,
    State: StateInstance + From<T>,
    Value: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let present = self
                    .tuple
                    .iter()
                    .cloned()
                    .map(State::from)
                    .filter_map(Value::opt_cast_from)
                    .any(|item| item == key);

                Ok(present.into())
            })
        }))
    }
}

impl<'a, State, T> From<&'a Tuple<T>> for ContainsHandler<'a, State, T> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        Self {
            tuple,
            state: PhantomData,
        }
    }
}

struct ForEachHandler<I> {
    source: I,
}

impl<'a, Txn, State, I, T> Handler<'a, Txn, State> for ForEachHandler<I>
where
    I: IntoIterator<Item = T> + Send + 'a,
    I::IntoIter: Send + 'a,
    T: 'a,
    State: From<T>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                // let op: Closure = params.require(&label("op").into())?;
                // params.expect_empty()?;
                //
                // stream::iter(self.source)
                //     .map(|args| op.clone().call(txn, State::from(args)))
                //     .buffer_unordered(num_cpus::get())
                //     .try_fold((), |(), _| future::ready(Ok(())))
                //     .await?;
                //
                // Ok(State::default())
                Err(not_implemented!("generic for each"))
            })
        }))
    }
}

struct MapCopyHandler<'a, FE, T> {
    map: &'a Map<T>,
    phantom: PhantomData<FE>,
}

impl<'a, FE, T> Handler<'a, T::Txn, T::State> for MapCopyHandler<'a, FE, T>
where
    FE: ThreadSafe,
    T: Route<FE> + Clone + Send + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, T::Txn, T::State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                let path = TCPathBuf::from(COPY);
                let mut copies: FuturesOrdered<_> = self
                    .map
                    .iter()
                    .map(|(key, item)| {
                        let key = key.clone();
                        let result = Public::get(item, txn, &path, Value::default());
                        result.map_ok(move |state| (key, state))
                    })
                    .collect();

                let mut copy = Map::new();
                while let Some((key, item)) = copies.try_next().await? {
                    copy.insert(key, item);
                }

                Ok(copy.into())
            })
        }))
    }
}

impl<'a, FE, T> From<&'a Map<T>> for MapCopyHandler<'a, FE, T> {
    fn from(map: &'a Map<T>) -> Self {
        Self {
            map,
            phantom: PhantomData,
        }
    }
}

struct MapEqHandler<State, T> {
    map: Map<T>,
    state: PhantomData<State>,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for MapEqHandler<State, T>
where
    State: StateInstance + From<T>,
    T: fmt::Debug + Send + Sync + 'a,
    Map<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
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

                let eq = self
                    .map
                    .into_iter()
                    .zip(other)
                    .map(|((this_id, this), (that_id, that))| {
                        if this_id != that_id {
                            return Ok(false);
                        }

                        let this = State::from(this);
                        let this =
                            Value::try_cast_from(this, |s| TCError::unexpected(s, "a Value"))?;

                        let that =
                            Value::try_cast_from(that, |s| TCError::unexpected(s, "a Value"))?;

                        Ok(this == that)
                    })
                    .collect::<TCResult<Vec<bool>>>()?;

                Ok(eq.into_iter().all(|eq| eq).into())
            })
        }))
    }
}

impl<State, T> From<Map<T>> for MapEqHandler<State, T> {
    fn from(map: Map<T>) -> Self {
        Self {
            map,
            state: PhantomData,
        }
    }
}

struct EqTupleHandler<State, T> {
    tuple: Tuple<T>,
    state: PhantomData<State>,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for EqTupleHandler<State, T>
where
    T: fmt::Debug + Send + Sync + 'a,
    State: StateInstance + From<T>,
    Tuple<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
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

                let eq = self
                    .tuple
                    .into_iter()
                    .zip(other)
                    .map(|(this, that)| {
                        let this = State::from(this);

                        let this =
                            Value::try_cast_from(this, |s| TCError::unexpected(s, "a Value"))?;

                        let that =
                            Value::try_cast_from(that, |s| TCError::unexpected(s, "a Value"))?;

                        Ok(this == that)
                    })
                    .collect::<TCResult<Vec<bool>>>()?;

                Ok(eq.into_iter().all(|eq| eq).into())
            })
        }))
    }
}

impl<State, T> From<Tuple<T>> for EqTupleHandler<State, T> {
    fn from(tuple: Tuple<T>) -> Self {
        Self {
            tuple,
            state: PhantomData,
        }
    }
}

impl<'a, State, T: Clone> From<&'a Tuple<T>> for AppendHandler<'a, State, T> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        Self {
            tuple,
            state: PhantomData,
        }
    }
}

struct MapHandler<'a, T: Clone> {
    map: &'a Map<T>,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for MapHandler<'a, T>
where
    State: StateInstance + From<T> + From<Map<T>>,
    T: Instance + Clone,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    Ok(State::from(self.map.clone()))
                } else {
                    let key = Id::try_cast_from(key, |v| TCError::unexpected(v, "an Id"))?;
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

impl<FE, T> Route<FE> for Map<T>
where
    FE: ThreadSafe,
    T: Instance + Route<FE> + Clone + fmt::Debug,
    T::State: From<T> + From<Map<T>> + From<Number>,
    Map<T::State>: TryCastFrom<T::State>,
    Tuple<T::State>: TryCastFrom<T::State>,
    Value: TryCastFrom<T::State>,
{
    type Txn = T::Txn;
    type State = T::State;

    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, T::Txn, T::State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(MapHandler { map: self }))
        } else if let Some(state) = self.deref().get(&path[0]) {
            state.route(&path[1..])
        } else if path.len() == 1 {
            match path[0].as_str() {
                "copy" => Some(Box::new(MapCopyHandler::from(self))),
                "eq" => Some(Box::new(MapEqHandler::from(self.clone()))),
                "len" => Some(Box::new(AttributeHandler::from(Number::UInt(UInt::U64(
                    self.len() as u64,
                ))))),
                _ => None,
            }
        } else {
            None
        }
    }
}

struct TupleCopyHandler<'a, FE, T> {
    tuple: &'a Tuple<T>,
    phantom: PhantomData<FE>,
}

impl<'a, FE, T> Handler<'a, T::Txn, T::State> for TupleCopyHandler<'a, FE, T>
where
    FE: ThreadSafe,
    T: Route<FE> + Clone + Send + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, T::Txn, T::State>>
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

                Ok(T::State::from(Tuple::from(copy)))
            })
        }))
    }
}

impl<'a, FE, T> From<&'a Tuple<T>> for TupleCopyHandler<'a, FE, T> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        Self {
            tuple,
            phantom: PhantomData,
        }
    }
}

struct TupleFoldHandler<'a, T> {
    tuple: &'a Tuple<T>,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for TupleFoldHandler<'a, T>
where
    T: Clone + Send + Sync + 'a,
    State: From<T>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                // let item_name: Id = params.require(&label("item_name").into())?;
                // let op: Closure = params.require(&label("op").into())?;
                // let mut state: State = params.require(&label("value").into())?;
                // params.expect_empty()?;
                //
                // for item in self.tuple.iter().cloned() {
                //     let mut params: Map<State> = state.try_into()?;
                //     params.insert(item_name.clone(), item.into());
                //     state = op.clone().call(txn, params.into()).await?;
                // }
                //
                // Ok(state.into())
                Err(not_implemented!("tuple fold"))
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

impl<'a, Txn, State, I> Handler<'a, Txn, State> for MapOpHandler<I>
where
    I: IntoIterator + Send + 'a,
    I::IntoIter: Send + 'a,
    State: From<I::Item>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                // let op: Closure = params.require(&label("op").into())?;
                //
                // let mut tuple = Vec::with_capacity(self.len);
                // let mut mapped = stream::iter(self.items)
                //     .map(State::from)
                //     .map(|state| op.clone().call(txn, state))
                //     .buffered(num_cpus::get());
                //
                // while let Some(item) = mapped.try_next().await? {
                //     tuple.push(item);
                // }
                //
                // Ok(State::Tuple(tuple.into()))
                Err(not_implemented!("generic map op"))
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

impl<'a, Txn, State, T> Handler<'a, Txn, State> for TupleHandler<'a, T>
where
    State: StateInstance + From<T> + From<Tuple<T>>,
    T: Instance + Clone,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let len = self.tuple.len() as i64;

                match key {
                    Value::None => Ok(State::from(self.tuple.clone())),
                    Value::Tuple(range) if range.matches::<(Bound<Value>, Bound<Value>)>() => {
                        let range: (Bound<Value>, Bound<Value>) =
                            range.opt_cast_into().expect("range");

                        let (start, end) = cast_range(range, len)?;
                        let slice = self.tuple[start..end]
                            .iter()
                            .cloned()
                            .collect::<Tuple<State>>();

                        Ok(State::from(slice))
                    }
                    i if i.matches::<Number>() => {
                        let i = Number::opt_cast_from(i).expect("tuple index");
                        let i = i64::cast_from(i);
                        let i = if i < 0 { len + i } else { i };

                        self.tuple
                            .get(i as usize)
                            .cloned()
                            .map(State::from)
                            .ok_or_else(|| TCError::not_found(format!("no such index: {}", i)))
                    }
                    other => Err(TCError::unexpected(other, "a tuple index")),
                }
            })
        }))
    }
}

struct ZipHandler<'a, T: Clone> {
    keys: &'a Tuple<T>,
}

impl<'a, Txn, State, T> Handler<'a, Txn, State> for ZipHandler<'a, T>
where
    State: StateInstance + From<T>,
    T: Clone + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let values: Tuple<Value> =
                    key.try_cast_into(|v| bad_request!("invalid values for Tuple/zip: {}", v))?;

                if self.keys.len() != values.len() {
                    return Err(bad_request!(
                        "cannot zip {} keys with {} values",
                        self.keys.len(),
                        values.len()
                    ));
                }

                let zipped =
                    self.keys
                        .iter()
                        .cloned()
                        .zip(values.into_iter())
                        .map(|(key, value)| {
                            let key = State::from(key);
                            let value = State::from(value);
                            State::from(Tuple::from(vec![key, value]))
                        });

                Ok(State::from(zipped.collect::<Tuple<State>>()))
            })
        }))
    }
}

impl<'a, T: Clone> From<&'a Tuple<T>> for ZipHandler<'a, T> {
    fn from(keys: &'a Tuple<T>) -> Self {
        ZipHandler { keys }
    }
}

impl<FE, T> Route<FE> for Tuple<T>
where
    FE: ThreadSafe,
    T: Instance + Route<FE> + Clone + fmt::Debug + 'static,
    T::State: From<T> + From<Tuple<T>> + From<Number>,
    Tuple<T::State>: TryCastFrom<T::State>,
    Value: From<T::State>,
{
    type Txn = T::Txn;
    type State = T::State;

    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, T::Txn, T::State> + 'a>> {
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
                "contains" => Some(Box::new(ContainsHandler::from(self))),
                "copy" => Some(Box::new(TupleCopyHandler::from(self))),
                "fold" => Some(Box::new(TupleFoldHandler::from(self))),
                "for_each" => Some(Box::new(ForEachHandler {
                    source: self.clone(),
                })),
                "len" => Some(Box::new(AttributeHandler::from(Number::UInt(UInt::U64(
                    self.len() as u64,
                ))))),
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

struct CreateMapHandler;

impl<'a, Txn, State> Handler<'a, Txn, State> for CreateMapHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let value =
                    Tuple::<(Id, Value)>::try_cast_from(key, |v| TCError::unexpected(v, "a Map"))?;

                let map = value.into_iter().collect::<Map<State>>();

                Ok(State::from(map))
            })
        }))
    }
}

pub struct MapStatic<Txn, State> {
    phantom: PhantomData<(Txn, State)>,
}

impl<FE, Txn, State> Route<FE> for MapStatic<Txn, State>
where
    Txn: Transaction<FE>,
    State: StateInstance,
{
    type Txn = Txn;
    type State = State;

    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, Txn, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateMapHandler))
        } else {
            None
        }
    }
}

struct CreateTupleHandler;

impl<'a, Txn, State> Handler<'a, Txn, State> for CreateTupleHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let value: Tuple<Value> = key.try_into()?;
                Ok(State::from(value.into_iter().collect::<Tuple<State>>()))
            })
        }))
    }
}

struct CreateRangeHandler;

impl<'a, Txn, State> Handler<'a, Txn, State> for CreateRangeHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.matches::<(i64, i64, usize)>() {
                    let (start, stop, step): (i64, i64, usize) =
                        key.opt_cast_into().expect("range");

                    Ok(State::from(
                        (start..stop)
                            .step_by(step)
                            .into_iter()
                            .map(Number::from)
                            .collect::<Tuple<State>>(),
                    ))
                } else if key.matches::<(i64, i64)>() {
                    let (start, stop): (i64, i64) = key.opt_cast_into().expect("range");
                    Ok(State::from(
                        (start..stop)
                            .into_iter()
                            .map(Number::from)
                            .collect::<Tuple<State>>(),
                    ))
                } else if key.matches::<usize>() {
                    let stop: usize = key.opt_cast_into().expect("range stop");
                    Ok(State::from(
                        (0..stop as u64)
                            .into_iter()
                            .map(Number::from)
                            .collect::<Tuple<State>>(),
                    ))
                } else {
                    Err(TCError::unexpected(key, "a range"))
                }
            })
        }))
    }
}

struct ConcatenateHandler;

impl<'a, Txn, State> Handler<'a, Txn, State> for ConcatenateHandler
where
    State: StateInstance,
    Tuple<State>: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let l: Tuple<State> = params.require(&label("l").into())?;
                let r: Tuple<State> = params.require(&label("r").into())?;
                params.expect_empty()?;

                let mut concat = l.into_inner();
                concat.extend(r.into_inner());
                Ok(State::from(Tuple::from(concat)))
            })
        }))
    }
}

#[derive(Default)]
pub struct TupleStatic<Txn, State> {
    phantom: PhantomData<(Txn, State)>,
}

impl<FE, Txn, State> Route<FE> for TupleStatic<Txn, State>
where
    Txn: Transaction<FE>,
    State: StateInstance,
    Tuple<State>: TryCastFrom<State>,
{
    type Txn = Txn;
    type State = State;

    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, Txn, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateTupleHandler))
        } else if path.len() == 1 {
            match path[0].as_str() {
                "concatenate" => Some(Box::new(ConcatenateHandler)),
                "range" => Some(Box::new(CreateRangeHandler)),
                _ => None,
            }
        } else {
            None
        }
    }
}

#[inline]
fn cast_range(range: (Bound<Value>, Bound<Value>), len: i64) -> TCResult<(usize, usize)> {
    #[inline]
    fn as_i64(v: Value) -> i64 {
        let n = Number::opt_cast_from(v).expect("start index");
        n.cast_into()
    }

    let start = match range.0 {
        Bound::Included(v) if v.matches::<Number>() => {
            let start = as_i64(v);
            if start < 0 {
                len + start
            } else {
                start
            }
        }
        Bound::Excluded(v) if v.matches::<Number>() => {
            let start = as_i64(v);
            let start = if start < 0 { len + start } else { start };
            start + 1
        }
        Bound::Unbounded => 0,
        other => return Err(bad_request!("invalid start index {:?} for Tuple", other)),
    };

    let end = match range.1 {
        Bound::Included(v) if v.matches::<Number>() => {
            let end = as_i64(v);
            let end = if end < 0 { len + end } else { end };
            end + 1
        }
        Bound::Excluded(v) if v.matches::<Number>() => {
            let end = as_i64(v);
            if end < 0 {
                len + end
            } else {
                end
            }
        }
        Bound::Unbounded => len,
        other => return Err(bad_request!("invalid end index {:?} for Tuple", other)),
    };

    if start >= len {
        Err(bad_request!(
            "start index {} is out of bounds for Tuple with length {}",
            start,
            len
        ))
    } else if end > len {
        Err(bad_request!(
            "end index {} is out of bounds for Tuple with length {}",
            end,
            len
        ))
    } else if start > end {
        Err(bad_request!(
            "invalid range for Tuple: {}",
            Tuple::<i64>::from((start, end)),
        ))
    } else {
        Ok((start as usize, end as usize))
    }
}
