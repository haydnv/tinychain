use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;

use crate::chain::{Chain, ChainType};
use crate::collection::{Collection, CollectionType};
use crate::error;
use crate::general::{Map, TCResult, TryCastFrom, Tuple};
use crate::handler::*;
use crate::object::{Object, ObjectType};
use crate::scalar::{
    label, Link, MethodType, PathSegment, Scalar, ScalarType, TCPathBuf, Value, ValueType,
};

pub trait Class: Into<Link> + Clone + Eq + fmt::Display {
    type Instance: Instance;
}

pub trait NativeClass: Class {
    fn from_path(path: &[PathSegment]) -> TCResult<Self>;

    fn prefix() -> TCPathBuf;
}

pub trait Instance: Clone + Send + Sync {
    type Class: Class;

    fn class(&self) -> Self::Class;

    fn is_a(&self, dtype: Self::Class) -> bool {
        self.class() == dtype
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TCType {
    Chain(ChainType),
    Collection(CollectionType),
    Map,
    Object(ObjectType),
    Scalar(ScalarType),
    Tuple,
}

impl Class for TCType {
    type Instance = State;
}

impl NativeClass for TCType {
    fn from_path(path: &[PathSegment]) -> TCResult<TCType> {
        let suffix = Self::prefix().try_suffix(path)?;

        match suffix[0].as_str() {
            "chain" => ChainType::from_path(path).map(TCType::Chain),
            "collection" => CollectionType::from_path(path).map(TCType::Collection),
            "object" => ObjectType::from_path(path).map(TCType::Object),
            "op" | "tuple" | "value" => ScalarType::from_path(path).map(TCType::Scalar),
            other => Err(error::not_found(other)),
        }
    }

    fn prefix() -> TCPathBuf {
        label("sbin").into()
    }
}

impl From<ChainType> for TCType {
    fn from(ct: ChainType) -> TCType {
        TCType::Chain(ct)
    }
}

impl From<CollectionType> for TCType {
    fn from(ct: CollectionType) -> TCType {
        TCType::Collection(ct)
    }
}

impl From<ObjectType> for TCType {
    fn from(ot: ObjectType) -> TCType {
        TCType::Object(ot)
    }
}

impl From<ValueType> for TCType {
    fn from(vt: ValueType) -> TCType {
        TCType::Scalar(ScalarType::Value(vt))
    }
}

impl TryFrom<TCType> for ValueType {
    type Error = error::TCError;

    fn try_from(class: TCType) -> TCResult<ValueType> {
        match class {
            TCType::Scalar(ScalarType::Value(class)) => Ok(class),
            other => Err(error::bad_request("Expected ValueType, found", other)),
        }
    }
}

impl From<TCType> for Link {
    fn from(t: TCType) -> Link {
        match t {
            TCType::Chain(ct) => ct.into(),
            TCType::Collection(ct) => ct.into(),
            TCType::Map => ScalarType::Map.into(),
            TCType::Object(ot) => ot.into(),
            TCType::Scalar(st) => st.into(),
            TCType::Tuple => ScalarType::Tuple.into(),
        }
    }
}

impl fmt::Display for TCType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(ct) => write!(f, "{}", ct),
            Self::Collection(ct) => write!(f, "{}", ct),
            Self::Map => write!(f, "type Map"),
            Self::Object(ot) => write!(f, "{}", ot),
            Self::Scalar(st) => write!(f, "{}", st),
            Self::Tuple => write!(f, "type Tuple"),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Chain(Chain),
    Collection(Collection),
    Map(Map<State>),
    Object(Object),
    Scalar(Scalar),
    Tuple(Tuple<State>),
}

impl State {
    pub fn is_none(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_none(),
            _ => false,
        }
    }

    pub fn is_scalar(&self) -> bool {
        match self {
            Self::Scalar(_) => true,
            _ => false,
        }
    }
}

impl Instance for State {
    type Class = TCType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Chain(chain) => chain.class().into(),
            Self::Collection(collection) => collection.class().into(),
            Self::Map(_) => TCType::Map,
            Self::Object(object) => object.class().into(),
            Self::Scalar(scalar) => scalar.class().into(),
            Self::Tuple(_) => TCType::Tuple,
        }
    }
}

impl Route for State {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        match self {
            Self::Chain(chain) => chain.route(method, path),
            Self::Collection(collection) => collection.route(method, path),
            Self::Map(map) => map.route(method, path),
            Self::Object(object) => object.route(method, path),
            Self::Scalar(scalar) => scalar.route(method, path),
            Self::Tuple(tuple) => tuple.route(method, path),
        }
    }
}

impl Route for Map<State> {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if path.is_empty() {
            None
        } else if let Some(state) = self.deref().get(&path[0]) {
            match state {
                State::Scalar(Scalar::Op(op_def)) if path.len() == 1 => {
                    Some(op_def.handler(Some(self.clone().into())))
                }
                other => other.route(method, &path[1..]),
            }
        } else {
            None
        }
    }
}

impl Route for Tuple<State> {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if path.is_empty() {
            None
        } else if usize::can_cast_from(&path[0]) {
            let i = usize::opt_cast_from(path[0].clone()).unwrap();

            match self.deref().get(i) {
                Some(State::Scalar(Scalar::Op(op_def))) if path.len() == 1 => {
                    Some(op_def.handler(Some(self.clone().into())))
                }
                Some(other) => other.route(method, &path[1..]),
                None => None,
            }
        } else {
            None
        }
    }
}

impl From<Chain> for State {
    fn from(c: Chain) -> State {
        Self::Chain(c)
    }
}

impl From<Collection> for State {
    fn from(c: Collection) -> State {
        Self::Collection(c)
    }
}

impl From<Map<State>> for State {
    fn from(map: Map<State>) -> State {
        Self::Map(map)
    }
}

impl From<Object> for State {
    fn from(o: Object) -> State {
        State::Object(o)
    }
}

impl From<Scalar> for State {
    fn from(s: Scalar) -> State {
        Self::Scalar(s)
    }
}

impl From<Tuple<State>> for State {
    fn from(tuple: Tuple<State>) -> State {
        State::Tuple(tuple)
    }
}

impl From<Value> for State {
    fn from(v: Value) -> State {
        Self::Scalar(Scalar::from(v))
    }
}

impl From<()> for State {
    fn from(_: ()) -> State {
        Self::Scalar(Scalar::Value(Value::None))
    }
}

impl TryFrom<State> for Chain {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Chain> {
        match state {
            State::Chain(chain) => Ok(chain),
            other => Err(error::bad_request("Expected Chain but found", other)),
        }
    }
}

impl TryFrom<State> for Object {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Object> {
        match state {
            State::Object(object) => Ok(object),
            other => Err(error::bad_request("Expected Object but found", other)),
        }
    }
}

impl TryFrom<State> for Scalar {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Scalar> {
        match state {
            State::Scalar(scalar) => Ok(scalar),
            other => Err(error::bad_request("Expected Scalar but found", other)),
        }
    }
}

impl TryFrom<State> for Value {
    type Error = error::TCError;

    fn try_from(state: State) -> TCResult<Value> {
        let scalar = Scalar::try_from(state)?;
        scalar.try_into()
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Chain(c) => write!(f, "{}", c),
            Self::Collection(c) => write!(f, "{}", c),
            Self::Map(map) => write!(
                f,
                "{{{}}}",
                map.iter()
                    .map(|(k, v)| format!("{}: {}", k, v))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
            Self::Object(o) => write!(f, "{}", o),
            Self::Scalar(s) => write!(f, "{}", s),
            Self::Tuple(tuple) => write!(
                f,
                "({})",
                tuple
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}
