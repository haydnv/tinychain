//! Public API endpoints for a [`BTree`]

use std::iter::FromIterator;
use std::ops::Bound;

use freqfs::FileSave;
use futures::{TryFutureExt, TryStreamExt};
use log::debug;
use safecast::{AsType, Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::fs::{Dir, Persist};
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route, StateInstance,
};
use tc_transact::{fs, Transaction};
use tc_value::Value;
use tcgeneric::{Map, PathSegment, TCPath, Tuple};

use crate::Collection;

use super::schema::{BTreeSchema, Column};
use super::{BTree, BTreeFile, BTreeInstance, BTreeType, BTreeWrite, Key, Node, Range};

impl<State> Route<State> for BTreeType
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    State::FE: for<'a> FileSave<'a> + AsType<Node>,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if self == &Self::default() {
            Static.route(path)
        } else {
            None
        }
    }
}

struct CopyHandler;

impl<'a, State> Handler<'a, State> for CopyHandler
where
    State: StateInstance,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, _params| {
            Box::pin(async move {
                // let schema: Value = params.require("schema")?;
                // let schema = cast_into_schema(schema)?;
                //
                // let source: Collection<State::Txn, State::FE> =
                //     params.require("source")?;
                //
                // params.expect_empty()?;
                //
                // let txn_id = *txn.id();
                //
                // let store = {
                //     let cxt = txn.context().await?;
                //     let mut cxt = cxt.write().await;
                //     let (_, cache) = cxt.create_dir_unique()?;
                //     Dir::load(*txn.id(), cache).await?
                // };

                Err(not_implemented!("BTree copy"))
            })
        }))
    }
}

struct CreateHandler;

impl<'a, State> Handler<'a, State> for CreateHandler
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    State::FE: for<'b> FileSave<'b> + AsType<Node>,
    BTreeFile<State::Txn, State::FE>:
        fs::Persist<State::FE, Txn = State::Txn, Schema = BTreeSchema>,
    Collection<State::Txn, State::FE>: From<BTreeFile<State::Txn, State::FE>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, value| {
            Box::pin(async move {
                let schema = cast_into_schema(value)?;
                let store = {
                    let cxt = txn.context().await?;
                    let mut cxt = cxt.write().await;
                    let (_, cache) = cxt.create_dir_unique()?;
                    Dir::load(*txn.id(), cache).await?
                };

                BTreeFile::<State::Txn, State::FE>::create(*txn.id(), schema, store)
                    .map_ok(Collection::<State::Txn, State::FE>::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct BTreeHandler<'a, T> {
    btree: &'a T,
}

impl<'a, State, T> Handler<'a, State> for BTreeHandler<'a, T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    T: BTreeInstance + BTreeWrite,
    BTree<State::Txn, State::FE>: BTreeInstance + From<T::Slice> + TryCastFrom<State>,
    Collection<State::Txn, State::FE>: From<BTree<State::Txn, State::FE>>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, range| {
            Box::pin(async move {
                let range = cast_into_range(Scalar::Value(range))?;
                let slice = self.btree.clone().slice(range, false)?;
                Ok(Collection::BTree(slice.into()).into())
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(bad_request!(
                        "BTree::insert does not support an explicit key {key:?}"
                    ));
                }

                if value.matches::<BTree<State::Txn, State::FE>>() {
                    debug!("insert keys from {:?}", value);
                    let btree =
                        BTree::<State::Txn, State::FE>::opt_cast_from(value).expect("collection");

                    let keys = btree.keys(*txn.id()).await?;
                    self.btree.try_insert_from(*txn.id(), keys).await
                } else if value.matches::<Value>() {
                    debug!("insert key {:?}", value);
                    let value = Value::opt_cast_from(value).expect("value");
                    let value = value.try_cast_into(|v| TCError::unexpected(v, "a BTree key"))?;

                    self.btree.insert(*txn.id(), value).await
                } else {
                    Err(TCError::unexpected(value, "a BTree key"))
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let reverse = params.or_default("reverse")?;
                let range = params.or_default("range")?;
                let range = cast_into_range(range)?;
                let slice = self.btree.clone().slice(range, reverse)?;
                Ok(Collection::BTree(slice.into()).into())
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, State::Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, range| {
            Box::pin(async move {
                let range = cast_into_range(Scalar::Value(range))?;
                self.btree.clone().delete(*txn.id(), range).await
            })
        }))
    }
}

impl<'a, T> From<&'a T> for BTreeHandler<'a, T> {
    fn from(btree: &'a T) -> Self {
        Self { btree }
    }
}

struct CountHandler<'a, T> {
    btree: &'a T,
}

impl<'a, State, T> Handler<'a, State> for CountHandler<'a, T>
where
    State: StateInstance + From<u64>,
    T: BTreeInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    self.btree.count(*txn.id()).map_ok(State::from).await
                } else {
                    let range = cast_into_range(Scalar::Value(key))?;
                    let slice = self.btree.clone().slice(range, false)?;
                    slice.count(*txn.id()).map_ok(State::from).await
                }
            })
        }))
    }
}

impl<'a, T> From<&'a T> for CountHandler<'a, T> {
    fn from(btree: &'a T) -> Self {
        Self { btree }
    }
}

struct FirstHandler<'a, T> {
    btree: &'a T,
}

impl<'a, State: StateInstance, T: BTreeInstance> Handler<'a, State> for FirstHandler<'a, T> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(bad_request!("BTree::first does not accept a key {key:?}"));
                }

                let mut keys = self.btree.clone().keys(*txn.id()).await?;
                if let Some(values) = keys.try_next().await? {
                    let names = self.btree.schema().iter().map(|col| col.name()).cloned();
                    Ok(
                        Map::<State>::from_iter(names.zip(values.into_iter().map(State::from)))
                            .into(),
                    )
                } else {
                    Err(TCError::not_found("this BTree is empty"))
                }
            })
        }))
    }
}

impl<'a, T> From<&'a T> for FirstHandler<'a, T> {
    fn from(btree: &'a T) -> Self {
        Self { btree }
    }
}

struct ReverseHandler<T> {
    btree: T,
}

impl<'a, State, T> Handler<'a, State> for ReverseHandler<T>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    T: BTreeInstance + 'a,
    BTree<State::Txn, State::FE>: From<<T as BTreeInstance>::Slice>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(bad_request!("BTree::reverse does not accept a key {key:?}"));
                }

                let reversed = self.btree.slice(Range::default(), true)?;
                Ok(Collection::from(BTree::from(reversed)).into())
            })
        }))
    }
}

impl<T> From<T> for ReverseHandler<T> {
    fn from(btree: T) -> Self {
        Self { btree }
    }
}

impl<State> Route<State> for BTree<State::Txn, State::FE>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<u64>,
    State::FE: AsType<Node>,
    BTree<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("BTree::route {}", TCPath::from(path));
        route(self, path)
    }
}

impl<State> Route<State> for BTreeFile<State::Txn, State::FE>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<u64>,
    State::FE: AsType<Node>,
    BTree<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self, path)
    }
}

#[inline]
fn route<'a, State, T>(
    btree: &'a T,
    path: &'a [PathSegment],
) -> Option<Box<dyn Handler<'a, State> + 'a>>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<u64>,
    State::FE: AsType<Node>,
    T: BTreeInstance + BTreeWrite + 'a,
    BTree<State::Txn, State::FE>: BTreeInstance + From<T> + From<T::Slice> + TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    if path.is_empty() {
        Some(Box::new(BTreeHandler::from(btree)))
    } else if path.len() == 1 {
        match path[0].as_str() {
            "count" => Some(Box::new(CountHandler::from(btree))),
            "first" => Some(Box::new(FirstHandler::from(btree))),
            "reverse" => Some(Box::new(ReverseHandler::from(btree.clone()))),
            _ => None,
        }
    } else {
        None
    }
}

pub struct Static;

impl<State> Route<State> for Static
where
    State: StateInstance + From<Collection<State::Txn, State::FE>>,
    State::FE: for<'a> FileSave<'a> + AsType<Node>,
    BTreeFile<State::Txn, State::FE>:
        fs::Persist<State::FE, Schema = BTreeSchema, Txn = State::Txn>,
    Collection<State::Txn, State::FE>: From<BTreeFile<State::Txn, State::FE>> + TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(CreateHandler))
        } else if path == &["copy_from"] {
            Some(Box::new(CopyHandler))
        } else {
            None
        }
    }
}

#[inline]
fn cast_into_range(scalar: Scalar) -> TCResult<Range> {
    let mut range = if scalar.is_none() {
        return Ok(Range::default());
    } else {
        Tuple::<Scalar>::try_cast_from(scalar, |s| bad_request!("invalid BTree selector: {:?}", s))?
    };

    if range
        .last()
        .expect("bounds")
        .matches::<(Bound<Value>, Bound<Value>)>()
    {
        let bounds: (Bound<Value>, Bound<Value>) = range
            .pop()
            .expect("bounds")
            .opt_cast_into()
            .expect("bounds");

        let prefix = range
            .into_iter()
            .map(|v| {
                Value::try_cast_from(v, |v| {
                    bad_request!("invalid value for BTree range: {:?}", v)
                })
            })
            .collect::<TCResult<Key>>()?;

        Ok(Range::with_bounds(prefix, bounds))
    } else {
        debug!("this is not a range: {:?}", range.last().expect("last"));

        let prefix = range
            .into_iter()
            .map(|v| {
                Value::try_cast_from(v, |v| {
                    bad_request!("invalid value for BTree range: {:?}", v)
                })
            })
            .collect::<TCResult<Key>>()?;

        Ok(Range::from_prefix(prefix))
    }
}

#[inline]
fn cast_into_schema(schema: Value) -> TCResult<BTreeSchema> {
    let columns = if let Value::Tuple(columns) = schema {
        columns
            .into_iter()
            .map(|col| {
                Column::try_cast_from(col, |v| bad_request!("invalid column schema: {:?}", v))
            })
            .collect::<TCResult<Vec<Column>>>()
    } else {
        Err(bad_request!("invalid BTree schema: {:?}", schema))
    }?;

    BTreeSchema::new(columns)
}
