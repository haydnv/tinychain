use async_trait::async_trait;
use destream::en;
use tc_error::{not_implemented, TCResult};

use tc_value::{Number, ToUrl, Value};

use tc_server::{RPCClient, Txn};
use tc_transact::public::{ClosureInstance, Handler, Route, StateInstance};
use tcgeneric::{path_label, Class, Instance, Map, NativeClass, PathSegment, TCPathBuf, Tuple};

#[derive(Clone, Debug, Default)]
pub struct Closure {}

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
    Value(Value),
}

impl Default for State {
    fn default() -> Self {
        Self::Value(Value::default())
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

impl Route<Self> for State {
    fn route<'a>(&'a self, _path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, Self> + 'a>> {
        None
    }
}

impl From<bool> for State {
    fn from(value: bool) -> Self {
        Self::Value(value.into())
    }
}

impl From<Closure> for State {
    fn from(closure: Closure) -> Self {
        Self::Closure(closure)
    }
}

impl From<Number> for State {
    fn from(n: Number) -> Self {
        Self::Value(n.into())
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

impl From<Value> for State {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl From<StateType> for State {
    fn from(class: StateType) -> Self {
        Self::Value(class.path().into())
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
