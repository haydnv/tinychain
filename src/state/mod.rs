use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::{self, Decoder, FromStream, MapAccess, SeqAccess, Visitor};
use destream::en::{Encoder, IntoStream, ToStream};
use futures::TryFutureExt;
use log::debug;
use safecast::{Match, TryCastFrom};

use generic::*;

pub use crate::scalar::*;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum StateType {
    Map,
    Scalar(ScalarType),
    Tuple,
}

impl Class for StateType {
    type Instance = State;
}

impl NativeClass for StateType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("StateType::from_path {}", TCPath::from(path));

        if path.is_empty() {
            None
        } else if &path[0] == "state" {
            if path.len() == 2 {
                match path[1].as_str() {
                    "map" => Some(Self::Map),
                    "tuple" => Some(Self::Tuple),
                    _ => None,
                }
            } else if path.len() > 2 && &path[1] == "scalar" {
                ScalarType::from_path(path).map(Self::Scalar)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::Map => path_label(&["state", "map"]).into(),
            Self::Scalar(st) => st.path(),
            Self::Tuple => path_label(&["state", "tuple"]).into(),
        }
    }
}

impl fmt::Display for StateType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Map => f.write_str("Map<Id, State>"),
            Self::Scalar(st) => fmt::Display::fmt(st, f),
            Self::Tuple => f.write_str("Tuple<State>"),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Map(Map<Self>),
    Scalar(Scalar),
    Tuple(Tuple<Self>),
}

impl State {
    fn is_none(&self) -> bool {
        match self {
            Self::Scalar(scalar) => scalar.is_none(),
            Self::Tuple(tuple) => tuple.is_empty(),
            _ => false,
        }
    }
}

impl Default for State {
    fn default() -> Self {
        Self::Scalar(Scalar::default())
    }
}

impl Instance for State {
    type Class = StateType;

    fn class(&self) -> StateType {
        match self {
            Self::Map(_) => StateType::Map,
            State::Scalar(scalar) => StateType::Scalar(scalar.class()),
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}

impl From<Scalar> for State {
    fn from(scalar: Scalar) -> State {
        State::Scalar(scalar)
    }
}

impl From<Value> for State {
    fn from(value: Value) -> State {
        State::Scalar(value.into())
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Map(map) => fmt::Display::fmt(map, f),
            Self::Scalar(scalar) => fmt::Display::fmt(scalar, f),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, f),
        }
    }
}

impl TryCastFrom<State> for Id {
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

impl TryCastFrom<State> for Value {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Map(_) => false,
            State::Scalar(scalar) => Self::can_cast_from(scalar),
            State::Tuple(tuple) => tuple.iter().all(Self::can_cast_from),
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Map(_) => None,
            State::Scalar(scalar) => Self::opt_cast_from(scalar),
            State::Tuple(tuple) => {
                let mut value = Vec::with_capacity(tuple.len());
                for item in tuple.into_iter() {
                    if let Some(v) = Self::opt_cast_from(item) {
                        value.push(v);
                    } else {
                        return None;
                    }
                }
                Some(Value::Tuple(value.into()).into())
            }
        }
    }
}

impl<T: TryCastFrom<State>> TryCastFrom<State> for (T,) {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<State>, T2: TryCastFrom<State>> TryCastFrom<State> for (T1, T2) {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T: Clone + TryCastFrom<State>> TryCastFrom<State> for Tuple<T> {
    fn can_cast_from(state: &State) -> bool {
        match state {
            State::Tuple(tuple) => tuple.iter().all(T::can_cast_from),
            _ => false,
        }
    }

    fn opt_cast_from(state: State) -> Option<Self> {
        match state {
            State::Tuple(source) => {
                let mut dest = Vec::with_capacity(source.len());
                for item in source.into_iter() {
                    if let Some(item) = T::opt_cast_from(item) {
                        dest.push(item);
                    } else {
                        return None;
                    }
                }

                Some(Tuple::from(dest))
            }
            _ => None,
        }
    }
}

#[derive(Default)]
struct StateVisitor {
    scalar: ScalarVisitor,
}

#[async_trait]
impl Visitor for StateVisitor {
    type Value = State;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tinychain State, e.g. 1 or [2] or \"three\" or {\"/state/scalar/value/number/complex\": [3.14, -1.414]")
    }

    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Self::Value, E> {
        self.scalar.visit_bool(b).map(State::Scalar)
    }

    fn visit_i8<E: de::Error>(self, i: i8) -> Result<Self::Value, E> {
        self.scalar.visit_i8(i).map(State::Scalar)
    }

    fn visit_i16<E: de::Error>(self, i: i16) -> Result<Self::Value, E> {
        self.scalar.visit_i16(i).map(State::Scalar)
    }

    fn visit_i32<E: de::Error>(self, i: i32) -> Result<Self::Value, E> {
        self.scalar.visit_i32(i).map(State::Scalar)
    }

    fn visit_i64<E: de::Error>(self, i: i64) -> Result<Self::Value, E> {
        self.scalar.visit_i64(i).map(State::Scalar)
    }

    fn visit_u8<E: de::Error>(self, u: u8) -> Result<Self::Value, E> {
        self.scalar.visit_u8(u).map(State::Scalar)
    }

    fn visit_u16<E: de::Error>(self, u: u16) -> Result<Self::Value, E> {
        self.scalar.visit_u16(u).map(State::Scalar)
    }

    fn visit_u32<E: de::Error>(self, u: u32) -> Result<Self::Value, E> {
        self.scalar.visit_u32(u).map(State::Scalar)
    }

    fn visit_u64<E: de::Error>(self, u: u64) -> Result<Self::Value, E> {
        self.scalar.visit_u64(u).map(State::Scalar)
    }

    fn visit_f32<E: de::Error>(self, f: f32) -> Result<Self::Value, E> {
        self.scalar.visit_f32(f).map(State::Scalar)
    }

    fn visit_f64<E: de::Error>(self, f: f64) -> Result<Self::Value, E> {
        self.scalar.visit_f64(f).map(State::Scalar)
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
        self.scalar.visit_string(s).map(State::Scalar)
    }

    fn visit_byte_buf<E: de::Error>(self, buf: Vec<u8>) -> Result<Self::Value, E> {
        self.scalar.visit_byte_buf(buf).map(State::Scalar)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        self.scalar.visit_unit().map(State::Scalar)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        self.scalar.visit_none().map(State::Scalar)
    }

    async fn visit_map<A: MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        if let Some(key) = access.next_key::<String>().await? {
            log::debug!("deserialize: key is {}", key);

            if let Ok(link) = Link::from_str(&key) {
                if link.host().is_none() {
                    if let Some(class) = StateType::from_path(link.path()) {
                        return match class {
                            StateType::Map => {
                                access
                                    .next_value::<HashMap<Id, State>>()
                                    .map_ok(Map::from)
                                    .map_ok(State::Map)
                                    .await
                            }
                            StateType::Scalar(st) => {
                                ScalarVisitor::visit_map_value(st, &mut access)
                                    .map_ok(State::Scalar)
                                    .await
                            }
                            StateType::Tuple => {
                                access
                                    .next_value::<Vec<State>>()
                                    .map_ok(Tuple::from)
                                    .map_ok(State::Tuple)
                                    .await
                            }
                        };
                    }
                } else {
                    let params: State = access.next_value().await?;
                    log::debug!("key is a Link, value is {}", params);

                    return if params.is_none() {
                        Ok(Value::Link(link).into())
                    } else if params.matches::<(Value, State)>() {
                        unimplemented!()
                    } else if params.matches::<(Value,)>() {
                        unimplemented!()
                    } else if let State::Map(_params) = params {
                        unimplemented!()
                    } else {
                        Err(de::Error::invalid_type(params, &"a Link or OpRef"))
                    };
                }
            }

            let key = Id::try_cast_from(key, |id| format!("invalid Id: {}", id))
                .map_err(de::Error::custom)?;

            let mut map = HashMap::new();
            let value = access.next_value().await?;
            map.insert(key, value);

            while let Some(key) = access.next_key().await? {
                let value = access.next_value().await?;
                map.insert(key, value);
            }

            Ok(State::Map(map.into()))
        } else {
            Ok(State::Map(Map::default()))
        }
    }

    async fn visit_seq<A: SeqAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut seq = if let Some(len) = access.size_hint() {
            Vec::with_capacity(len)
        } else {
            Vec::new()
        };

        while let Some(next) = access.next_element().await? {
            seq.push(next);
        }

        Ok(State::Tuple(seq.into()))
    }
}

#[async_trait]
impl FromStream for State {
    async fn from_stream<D: Decoder>(decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_any(StateVisitor::default()).await
    }
}

impl<'en> ToStream<'en> for State {
    fn to_stream<E: Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Map(map) => map.to_stream(encoder),
            Self::Scalar(scalar) => scalar.to_stream(encoder),
            Self::Tuple(tuple) => tuple.to_stream(encoder),
        }
    }
}

impl<'en> IntoStream<'en> for State {
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Map(map) => map.into_inner().into_stream(encoder),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
            Self::Tuple(tuple) => tuple.into_inner().into_stream(encoder),
        }
    }
}
