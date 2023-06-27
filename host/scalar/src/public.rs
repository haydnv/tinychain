use std::ops::Bound;

use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::public::generic::COPY;
use tc_transact::public::helpers::{AttributeHandler, EchoHandler};
use tc_transact::public::value;
use tc_transact::public::{
    ClosureInstance, DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route,
    StateInstance,
};
use tc_transact::RPCClient;
use tc_value::{Number, TCString, Value};
use tcgeneric::{Id, Map, PathSegment, TCPathBuf, Tuple};

use crate::{ClusterRef, Refer, Scalar, ScalarType};

impl<State> Route<State> for ScalarType
where
    State: StateInstance,
{
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        None
    }
}

struct ClusterHandler {
    path: TCPathBuf,
}

impl ClusterHandler {
    fn new(cluster: &ClusterRef, path: &[PathSegment]) -> Self {
        let mut cluster_path = cluster.path().clone();
        cluster_path.extend(path.into_iter().cloned());
        Self { path: cluster_path }
    }
}

impl<'a, State> Handler<'a, State> for ClusterHandler
where
    State: StateInstance,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| Box::pin(txn.get(self.path, key))))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(txn.put(self.path, key, value))
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(txn.post(self.path, params))
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, State::Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| Box::pin(txn.delete(self.path, key))))
    }
}

impl<State> Route<State> for ClusterRef
where
    State: StateInstance,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        Some(Box::new(ClusterHandler::new(self, path)))
    }
}

impl<State> Route<State> for Scalar
where
    State: StateInstance
        + Refer<State>
        + From<Scalar>
        + From<Tuple<Scalar>>
        + From<Map<Scalar>>
        + From<Value>
        + From<Tuple<Value>>
        + From<Number>,
    Box<dyn ClosureInstance<State>>: TryCastFrom<State>,
    Id: TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError> + TryCastFrom<State>,
    Number: TryCastFrom<State>,
    TCString: TryCastFrom<State>,
    Tuple<State>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path == &COPY[..] {
            return Some(Box::new(AttributeHandler::from(self.clone())));
        }

        match self {
            Self::Cluster(cluster) => cluster.route(path),
            Self::Map(map) => map.route(path),
            Self::Op(op_def) if path.is_empty() => Some(Box::new(op_def.clone())),
            Self::Range((start, end)) => {
                if path.is_empty() {
                    None
                } else {
                    match path[0].as_str() {
                        "start" => match start {
                            Bound::Included(value) => value.route(&path[1..]),
                            Bound::Excluded(value) => value.route(&path[1..]),
                            Bound::Unbounded => Value::None.route(&path[1..]),
                        },
                        "end" => match end {
                            Bound::Included(value) => value.route(&path[1..]),
                            Bound::Excluded(value) => value.route(&path[1..]),
                            Bound::Unbounded => Value::None.route(&path[1..]),
                        },
                        _ => None,
                    }
                }
            }
            Self::Ref(_) => None,
            Self::Value(value) => value.route(path),
            Self::Tuple(tuple) => tuple.route(path),
            _ => None,
        }
    }
}

pub struct Static;

impl<State> Route<State> for Static
where
    State: StateInstance,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(EchoHandler))
        } else if path[0] == value::PREFIX {
            value::Static.route(&path[1..])
        } else {
            None
        }
    }
}
