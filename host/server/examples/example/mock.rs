use std::collections::HashSet;

use async_trait::async_trait;
use destream::en;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::{OpDef, Refer, Scalar, Scope};
use tc_server::{RPCClient, Txn};
use tc_transact::public::{ClosureInstance, Handler, Public, Route, StateInstance, ToState};
use tc_value::{Number, ToUrl, Value};
use tcgeneric::{path_label, Class, Id, Instance, Map, NativeClass, PathSegment, TCPathBuf, Tuple};

#[derive(Clone, Debug)]
pub struct Closure {
    scope: Map<State>,
    op: OpDef,
}

impl From<(Map<State>, OpDef)> for Closure {
    fn from(closure: (Map<State>, OpDef)) -> Self {
        let (scope, op) = closure;
        Self { scope, op }
    }
}

impl TryCastFrom<State> for Closure {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Closure(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Closure(closure) => Some(closure),
            _ => None,
        }
    }
}

#[async_trait]
impl ClosureInstance<State> for Closure {
    async fn call(self: Box<Self>, _txn: Txn<State, CacheBlock>, _args: State) -> TCResult<State> {
        Err(not_implemented!("call a mock Closure"))
    }
}

#[derive(Copy, Clone, Debug)]
pub enum StateType {
    State,
}

impl Class for StateType {}

impl NativeClass for StateType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path == &["state"] {
            Some(Self::State)
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        path_label(&["state"]).into()
    }
}

#[derive(Clone, Debug)]
pub enum State {
    Closure(Closure),
    Map(Map<State>),
    Tuple(Tuple<State>),
    Scalar(Scalar),
}

impl Default for State {
    fn default() -> Self {
        Self::Scalar(Scalar::default())
    }
}

impl Instance for State {
    type Class = StateType;

    fn class(&self) -> Self::Class {
        StateType::State
    }
}

impl StateInstance for State {
    type FE = CacheBlock;
    type Txn = Txn<Self, CacheBlock>;
    type Closure = Closure;

    fn is_map(&self) -> bool {
        match self {
            Self::Map(_) => true,
            _ => false,
        }
    }

    fn is_tuple(&self) -> bool {
        match self {
            Self::Tuple(_) => true,
            _ => false,
        }
    }
}

#[async_trait]
impl Refer<State> for State {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        todo!()
    }

    fn is_conditional(&self) -> bool {
        todo!()
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        todo!()
    }

    fn is_ref(&self) -> bool {
        match self {
            Self::Closure(_) => true,
            Self::Map(map) => map.values().any(Self::is_ref),
            Self::Scalar(scalar) => Refer::<State>::is_ref(scalar),
            Self::Tuple(tuple) => tuple.into_iter().any(Self::is_ref),
        }
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        todo!()
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        todo!()
    }

    async fn resolve<'a, T>(
        self,
        scope: &'a Scope<'a, State, T>,
        txn: &'a Txn<State, CacheBlock>,
    ) -> TCResult<State>
    where
        T: ToState<State> + Public<State> + Instance,
    {
        match self {
            Self::Map(map) => {
                let mut resolved = Map::new();
                for (id, state) in map {
                    resolved.insert(id, state.resolve(scope, txn).await?);
                }
                Ok(resolved.into())
            }
            Self::Scalar(scalar) => scalar.resolve(scope, txn).await,
            Self::Tuple(tuple) => {
                let mut resolved = Tuple::with_capacity(tuple.len());
                for state in tuple {
                    resolved.push(state.resolve(scope, txn).await?);
                }
                Ok(resolved.into())
            }
            other => Ok(other),
        }
    }
}

impl PartialEq<Scalar> for State {
    fn eq(&self, other: &Scalar) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.eq(other),
            _ => false,
        }
    }
}

impl TryFrom<State> for Map<State> {
    type Error = TCError;

    fn try_from(state: State) -> Result<Self, Self::Error> {
        match state {
            State::Map(map) => Ok(map),
            other => Err(TCError::unexpected(other, "a Map")),
        }
    }
}

impl TryFrom<State> for Scalar {
    type Error = TCError;

    fn try_from(state: State) -> Result<Self, Self::Error> {
        match state {
            State::Scalar(scalar) => Ok(scalar),
            State::Tuple(tuple) => tuple
                .into_iter()
                .map(Scalar::try_from)
                .collect::<TCResult<_>>()
                .map(Scalar::Tuple),

            other => Err(TCError::unexpected(other, "a Scalar")),
        }
    }
}

impl TryFrom<State> for Value {
    type Error = TCError;

    fn try_from(state: State) -> Result<Self, Self::Error> {
        state
            .opt_cast_into()
            .ok_or_else(|| bad_request!("not a Value"))
    }
}

impl TryCastFrom<State> for Value {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.into_iter().all(Self::can_cast_from),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => scalar.opt_cast_into(),
            State::Tuple(tuple) => {
                let tuple = tuple
                    .into_iter()
                    .map(Self::opt_cast_from)
                    .collect::<Option<_>>()?;

                Some(Value::Tuple(tuple))
            }
            _ => None,
        }
    }
}

impl TryCastFrom<State> for bool {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            _ => None,
        }
    }
}

impl Route<Self> for State {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, Self> + 'a>> {
        None
    }
}

impl From<bool> for State {
    fn from(value: bool) -> Self {
        Self::Scalar(value.into())
    }
}

impl From<Closure> for State {
    fn from(closure: Closure) -> Self {
        Self::Closure(closure)
    }
}

impl From<Number> for State {
    fn from(n: Number) -> Self {
        Self::Scalar(n.into())
    }
}

impl From<Map<State>> for State {
    fn from(map: Map<State>) -> Self {
        Self::Map(map)
    }
}

impl From<Tuple<State>> for State {
    fn from(tuple: Tuple<State>) -> Self {
        Self::Tuple(tuple)
    }
}

impl From<Scalar> for State {
    fn from(scalar: Scalar) -> Self {
        Self::Scalar(scalar)
    }
}

impl From<Value> for State {
    fn from(value: Value) -> Self {
        Self::Scalar(value.into())
    }
}

impl From<StateType> for State {
    fn from(class: StateType) -> Self {
        Self::Scalar(Value::Link(class.path().into()).into())
    }
}

#[derive(Clone)]
pub enum CacheBlock {}

impl<'en> en::ToStream<'en> for CacheBlock {
    fn to_stream<En: en::Encoder<'en>>(&self, encoder: En) -> Result<En::Ok, En::Error> {
        en::IntoStream::into_stream((), encoder)
    }
}

#[derive(Default)]
pub struct Client {}

#[async_trait]
impl RPCClient<State> for Client {
    async fn get(&self, _link: ToUrl<'_>, _key: Value) -> TCResult<State> {
        Err(not_implemented!("mock RPCClient::get"))
    }

    async fn put(&self, _link: ToUrl<'_>, _key: Value, _value: State) -> TCResult<()> {
        Err(not_implemented!("mock RPCClient::get"))
    }

    async fn post(&self, _link: ToUrl<'_>, _params: Map<State>) -> TCResult<State> {
        Err(not_implemented!("mock RPCClient::get"))
    }

    async fn delete(&self, _link: ToUrl<'_>, _key: Value) -> TCResult<State> {
        Err(not_implemented!("mock RPCClient::get"))
    }
}
