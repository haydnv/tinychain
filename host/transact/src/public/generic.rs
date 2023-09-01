use std::convert::TryInto;
use std::fmt;
use std::iter::Cloned;
use std::marker::PhantomData;
use std::ops::{Bound, Deref};
use std::str::FromStr;

use futures::future::TryFutureExt;
use futures::stream::{self, FuturesOrdered, StreamExt, TryStreamExt};
use log::{debug, trace};
use safecast::*;

use tc_error::*;
use tc_value::{Number, UInt, Value};
use tcgeneric::{label, path_label, Id, Instance, Map, PathLabel, PathSegment, TCPathBuf, Tuple};

use super::helpers::AttributeHandler;
use super::{ClosureInstance, GetHandler, Handler, PostHandler, Public, Route, StateInstance};

type Closure<State> = Box<dyn ClosureInstance<State>>;

pub const COPY: PathLabel = path_label(&["copy"]);

struct AppendHandler<'a, T: Clone> {
    tuple: &'a Tuple<T>,
}

impl<'a, State, T> Handler<'a, State> for AppendHandler<'a, T>
where
    State: StateInstance + From<T>,
    T: fmt::Debug + Clone + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

struct ContainsHandler<'a, T> {
    tuple: &'a Tuple<T>,
}

impl<'a, State, T> Handler<'a, State> for ContainsHandler<'a, T>
where
    State: StateInstance + From<T>,
    T: Clone + Send + Sync,
    Value: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<'a, T> From<&'a Tuple<T>> for ContainsHandler<'a, T> {
    fn from(tuple: &'a Tuple<T>) -> Self {
        Self { tuple }
    }
}

struct ForEachHandler<I> {
    source: I,
}

impl<'a, State, I, T> Handler<'a, State> for ForEachHandler<I>
where
    State: StateInstance + From<T>,
    I: IntoIterator<Item = T> + Send + 'a,
    I::IntoIter: Send + 'a,
    T: 'a,
    Closure<State>: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let op: State = params.require(&label("op").into())?;
                params.expect_empty()?;

                let mut for_each = stream::iter(self.source)
                    .map(move |args| {
                        Closure::try_cast_from(op.clone(), |state| {
                            bad_request!("not a closure: {state:?}")
                        })
                        .map(|op| (op, args))
                    })
                    .and_then(|(op, args)| op.call(txn.clone(), State::from(args)));

                while let Some(_step) = for_each.try_next().await? {
                    // no-op
                }

                Ok(State::default())
            })
        }))
    }
}

impl<I> From<I> for ForEachHandler<I> {
    fn from(source: I) -> Self {
        Self { source }
    }
}

struct MapCopyHandler<'a, T> {
    map: &'a Map<T>,
}

impl<'a, State, T> Handler<'a, State> for MapCopyHandler<'a, T>
where
    State: StateInstance,
    T: Route<State> + Clone + Send + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<'a, T> From<&'a Map<T>> for MapCopyHandler<'a, T> {
    fn from(map: &'a Map<T>) -> Self {
        Self { map }
    }
}

struct MapEqHandler<State, T> {
    map: Map<T>,
    state: PhantomData<State>,
}

impl<'a, State, T> Handler<'a, State> for MapEqHandler<State, T>
where
    State: StateInstance + From<T>,
    T: fmt::Debug + Send + Sync + 'a,
    Map<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
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

struct EqTupleHandler<T> {
    tuple: Tuple<T>,
}

impl<'a, State, T> Handler<'a, State> for EqTupleHandler<T>
where
    T: fmt::Debug + Send + Sync + 'a,
    State: StateInstance + From<T>,
    Tuple<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
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

impl<'a, State, T> Handler<'a, State> for MapHandler<'a, T>
where
    State: StateInstance + From<T> + From<Map<T>>,
    T: Instance + Clone,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<State, T> Route<State> for Map<T>
where
    State: StateInstance + From<T> + From<Map<T>> + From<Number>,
    T: Instance + Route<State> + Clone + fmt::Debug,
    Map<State>: TryCastFrom<State>,
    Tuple<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
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

struct TupleCopyHandler<'a, T> {
    tuple: &'a Tuple<T>,
}

impl<'a, State, T> Handler<'a, State> for TupleCopyHandler<'a, T>
where
    State: StateInstance,
    T: Route<State> + Clone + Send + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

                Ok(State::from(Tuple::from(copy)))
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

impl<'a, State, T> Handler<'a, State> for TupleFoldHandler<'a, T>
where
    State: StateInstance + From<T>,
    T: Clone + Send + Sync + 'a,
    Closure<State>: TryCastFrom<State>,
    Id: TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let item_name: Id = params.require(&label("item_name").into())?;
                let op: State = params.require(&label("op").into())?;
                let mut state: State = params.require(&label("value").into())?;
                params.expect_empty()?;

                for item in self.tuple.iter().cloned() {
                    let mut params: Map<State> = state.try_into()?;
                    params.insert(item_name.clone(), item.into());

                    let op: Closure<State> = op
                        .clone()
                        .try_cast_into(|state| bad_request!("not a closure: {state:?}"))?;

                    state = op.call(txn.clone(), params.into()).await?;
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

impl<'a, State, I> Handler<'a, State> for MapOpHandler<I>
where
    State: StateInstance + From<I::Item>,
    I: IntoIterator + Send + 'a,
    I::IntoIter: Send + 'a,
    Closure<State>: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let op: State = params.require(&label("op").into())?;

                let mut tuple = Vec::with_capacity(self.len);
                let mut mapped = stream::iter(self.items)
                    .map(State::from)
                    .map(|state| {
                        Closure::try_cast_from(op.clone(), |state| {
                            bad_request!("not a closure: {state:?}")
                        })
                        .map(|op| op.call(txn.clone(), state))
                    })
                    .try_buffered(num_cpus::get());

                while let Some(item) = mapped.try_next().await? {
                    tuple.push(item);
                }

                Ok(State::from(Tuple::from(tuple)))
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

impl<'a, State, T> Handler<'a, State> for TupleHandler<'a, T>
where
    State: StateInstance + From<T> + From<Tuple<T>>,
    T: Instance + Clone + fmt::Debug,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                debug!("get {key:?} from {this:?}", this = self.tuple);

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

impl<'a, State, T> Handler<'a, State> for ZipHandler<'a, T>
where
    State: StateInstance + From<T>,
    T: Clone + Send + Sync,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<State, T> Route<State> for Tuple<T>
where
    State: StateInstance + From<T> + From<Tuple<T>>,
    T: Instance + Route<State> + Clone + fmt::Debug + 'static,
    Closure<State>: TryCastFrom<State>,
    Id: TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
    Tuple<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
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
                "for_each" => Some(Box::new(ForEachHandler::from(self.clone()))),
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

impl<'a, State: StateInstance> Handler<'a, State> for CreateMapHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

pub struct MapStatic;

impl<State: StateInstance> Route<State> for MapStatic {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateMapHandler))
        } else {
            None
        }
    }
}

struct CreateTupleHandler;

impl<'a, State> Handler<'a, State> for CreateTupleHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

impl<'a, State> Handler<'a, State> for CreateRangeHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                debug!("construct range Tuple from {key:?}");

                let (start, stop, step) = if key.matches::<(i64, i64, usize)>() {
                    key.opt_cast_into().expect("range")
                } else if key.matches::<(i64, i64)>() {
                    let (start, stop) = key.opt_cast_into().expect("range");
                    (start, stop, 1)
                } else if key.matches::<usize>() {
                    let stop = key.opt_cast_into().expect("range end");
                    (0, stop, 1)
                } else {
                    return Err(bad_request!("invalid range: {key}"));
                };

                let tuple = if start <= stop {
                    let range = (start..stop).step_by(step).into_iter();
                    State::from(range.map(Number::from).collect::<Tuple<State>>())
                } else {
                    let range = (stop..start).step_by(step).into_iter().rev();
                    State::from(range.map(Number::from).collect::<Tuple<State>>())
                };

                trace!("range is {tuple:?}");

                Ok(tuple)
            })
        }))
    }
}

struct ConcatenateHandler;

impl<'a, State> Handler<'a, State> for ConcatenateHandler
where
    State: StateInstance,
    Tuple<State>: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
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
pub struct TupleStatic;

impl<State> Route<State> for TupleStatic
where
    State: StateInstance,
    Tuple<State>: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
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
    trace!("construct range from bounds {range:?} within {len}");

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
