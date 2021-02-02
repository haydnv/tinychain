use std::collections::HashSet;
use std::fmt;
use std::ops::Deref;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::{self, Decoder, FromStream};
use destream::en::{EncodeMap, Encoder, IntoStream, ToStream};
use futures::TryFutureExt;
use safecast::{CastFrom, Match, TryCastFrom, TryCastInto};

use error::*;
use generic::*;

use crate::scalar::{Link, Scalar, Value};
use crate::state::State;
use crate::txn::Txn;

use super::{IdRef, Refer, TCRef};

const PREFIX: PathLabel = path_label(&["state", "scalar", "ref", "op"]);

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpRefType {
    Get,
    Put,
    Post,
    Delete,
}

impl Class for OpRefType {
    type Instance = OpRef;
}

impl NativeClass for OpRefType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 5 && &path[..4] == &PREFIX[..] {
            match path[4].as_str() {
                "get" => Some(Self::Get),
                "put" => Some(Self::Put),
                "post" => Some(Self::Post),
                "delete" => Some(Self::Delete),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let suffix = match self {
            Self::Get => "get",
            Self::Put => "put",
            Self::Post => "post",
            Self::Delete => "delete",
        };

        TCPathBuf::from(PREFIX).append(label(suffix))
    }
}

impl fmt::Display for OpRefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "GET Op ref"),
            Self::Put => write!(f, "PUT Op ref"),
            Self::Post => write!(f, "POST Op ref"),
            Self::Delete => write!(f, "DELETE Op ref"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Subject {
    Link(Link),
    Ref(IdRef),
}

impl FromStr for Subject {
    type Err = TCError;

    fn from_str(s: &str) -> TCResult<Self> {
        if s.starts_with('$') {
            IdRef::from_str(s).map(Self::Ref)
        } else {
            Link::from_str(s).map(Self::Link)
        }
    }
}

#[async_trait]
impl FromStream for Subject {
    type Context = ();

    async fn from_stream<D: Decoder>(context: (), d: &mut D) -> Result<Self, D::Error> {
        let subject = String::from_stream(context, d).await?;
        Subject::from_str(&subject).map_err(de::Error::custom)
    }
}

impl<'en> ToStream<'en> for Subject {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Link(link) => link.to_stream(e),
            Self::Ref(id_ref) => id_ref.to_stream(e),
        }
    }
}

impl<'en> IntoStream<'en> for Subject {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Link(link) => link.into_stream(e),
            Self::Ref(id_ref) => id_ref.into_stream(e),
        }
    }
}

impl TryCastFrom<Value> for Subject {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Link(_) => true,
            Value::String(s) => IdRef::from_str(s).is_ok() || Link::from_str(s).is_ok(),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Link(link) => Some(Self::Link(link)),
            Value::String(s) => {
                if let Ok(id_ref) = IdRef::from_str(&s) {
                    Some(Self::Ref(id_ref))
                } else if let Ok(link) = Link::from_str(&s) {
                    Some(Self::Link(link))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Subject {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => match &**tc_ref {
                TCRef::Id(_) => true,
                _ => false,
            },
            Scalar::Value(value) => Self::can_cast_from(value),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Some(Self::Ref(id_ref)),
                _ => None,
            },
            Scalar::Value(value) => Self::opt_cast_from(value),
            _ => None,
        }
    }
}

impl From<Link> for Subject {
    fn from(link: Link) -> Subject {
        Subject::Link(link)
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Link(link) => fmt::Display::fmt(link, f),
            Self::Ref(id_ref) => fmt::Display::fmt(id_ref, f),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Key {
    Ref(IdRef),
    Value(Value),
}

#[async_trait]
impl Refer for Key {
    fn requires(&self, deps: &mut HashSet<Id>) {
        if let Self::Ref(id_ref) = self {
            deps.insert(id_ref.id().clone());
        }
    }

    async fn resolve(self, context: &Map<State>, txn: &Txn) -> TCResult<State> {
        match self {
            Self::Ref(id_ref) => id_ref.resolve(context, txn).await,
            Self::Value(value) => Ok(State::from(value)),
        }
    }
}

impl CastFrom<Value> for Key {
    fn cast_from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl TryCastFrom<Scalar> for Key {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => match &**tc_ref {
                TCRef::Id(_) => true,
                _ => false,
            },
            other => Value::can_cast_from(other),
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Some(Self::Ref(id_ref)),
                _ => None,
            },
            other => {
                if let Some(value) = Value::opt_cast_from(other) {
                    Some(Self::Value(value))
                } else {
                    None
                }
            }
        }
    }
}

#[async_trait]
impl FromStream for Key {
    type Context = ();

    async fn from_stream<D: Decoder>(context: (), d: &mut D) -> Result<Self, D::Error> {
        match Scalar::from_stream(context, d).await? {
            Scalar::Value(value) => Ok(Self::Value(value)),
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Ok(Self::Ref(id_ref)),
                other => Err(de::Error::invalid_type(other, &"IdRef")),
            },
            other => Err(de::Error::invalid_type(other, &"a Value or IdRef")),
        }
    }
}

impl<'en> ToStream<'en> for Key {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Key::Ref(id_ref) => id_ref.to_stream(e),
            Key::Value(value) => value.to_stream(e),
        }
    }
}

impl<'en> IntoStream<'en> for Key {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Key::Ref(id_ref) => id_ref.into_stream(e),
            Key::Value(value) => value.into_stream(e),
        }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Key::Ref(tc_ref) => write!(f, "{}", tc_ref),
            Key::Value(value) => write!(f, "{}", value),
        }
    }
}

pub type GetRef = (Subject, Key);
pub type PutRef = (Subject, Key, Scalar);
pub type PostRef = (Subject, Map<Scalar>);
pub type DeleteRef = (Subject, Key);

#[derive(Clone, Eq, PartialEq)]
pub enum OpRef {
    Get(GetRef),
    Put(PutRef),
    Post(PostRef),
    Delete(DeleteRef),
}

impl Instance for OpRef {
    type Class = OpRefType;

    fn class(&self) -> OpRefType {
        use OpRefType as ORT;
        match self {
            Self::Get(_) => ORT::Get,
            Self::Put(_) => ORT::Put,
            Self::Post(_) => ORT::Post,
            Self::Delete(_) => ORT::Delete,
        }
    }
}

#[async_trait]
impl Refer for OpRef {
    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            OpRef::Get((_path, key)) => {
                if let Key::Ref(id_ref) = key {
                    deps.insert(id_ref.id().clone());
                }
            }
            OpRef::Put((_path, key, value)) => {
                if let Key::Ref(id_ref) = key {
                    deps.insert(id_ref.id().clone());
                }

                if let Scalar::Ref(tc_ref) = value {
                    tc_ref.requires(deps);
                }
            }
            OpRef::Post((_path, params)) => {
                for provider in params.values() {
                    if let Scalar::Ref(tc_ref) = provider {
                        tc_ref.requires(deps);
                    }
                }
            }
            OpRef::Delete((_path, key)) => {
                if let Key::Ref(id_ref) = key {
                    deps.insert(id_ref.id().clone());
                }
            }
        }
    }

    async fn resolve(self, _context: &Map<State>, _txn: &Txn) -> TCResult<State> {
        Err(TCError::not_implemented("OpRef::resolve"))
    }
}

pub struct OpRefVisitor;

impl OpRefVisitor {
    pub async fn visit_map_value<A: de::MapAccess>(
        class: OpRefType,
        access: &mut A,
    ) -> Result<OpRef, A::Error> {
        use OpRefType as ORT;

        match class {
            ORT::Get => access.next_value(()).map_ok(OpRef::Get).await,
            ORT::Put => access.next_value(()).map_ok(OpRef::Put).await,
            ORT::Post => access.next_value(()).map_ok(OpRef::Post).await,
            ORT::Delete => access.next_value(()).map_ok(OpRef::Delete).await,
        }
    }

    pub fn visit_ref_value<E: de::Error>(subject: Subject, params: Scalar) -> Result<OpRef, E> {
        match params {
            Scalar::Map(params) => Ok(OpRef::Post((subject, params))),
            Scalar::Tuple(params) if params.matches::<(Value, Scalar)>() => {
                let (key, value) = params.opt_cast_into().unwrap();
                Ok(OpRef::Put((subject, key, value)))
            }
            Scalar::Tuple(params) if params.matches::<(Value,)>() => {
                let (key,) = params.opt_cast_into().unwrap();
                Ok(OpRef::Get((subject, key)))
            }
            other => Err(de::Error::invalid_type(other, &"OpRef")),
        }
    }
}

#[async_trait]
impl de::Visitor for OpRefVisitor {
    type Value = OpRef;

    fn expecting() -> &'static str {
        "an OpRef, e.g. {\"$subject\": [\"key\"]}"
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let subject = access
            .next_key::<Subject>(())
            .await?
            .ok_or_else(|| de::Error::custom("expected OpRef, found empty map"))?;

        if let Subject::Link(link) = &subject {
            if link.host().is_none() {
                if let Some(class) = OpRefType::from_path(link.path()) {
                    return Self::visit_map_value(class, &mut access).await;
                }
            }
        }

        let params = access.next_value(()).await?;
        Self::visit_ref_value(subject, params)
    }
}

#[async_trait]
impl FromStream for OpRef {
    type Context = ();

    async fn from_stream<D: Decoder>(_: (), d: &mut D) -> Result<Self, D::Error> {
        d.decode_map(OpRefVisitor).await
    }
}

impl<'en> ToStream<'en> for OpRef {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        let mut map = e.encode_map(Some(1))?;

        match self {
            OpRef::Get((path, key)) => map.encode_entry(path.to_string(), key)?,
            OpRef::Put((path, key, value)) => map.encode_entry(path.to_string(), (key, value))?,
            OpRef::Post((path, data)) => map.encode_entry(path.to_string(), data.deref())?,
            OpRef::Delete((path, key)) => {
                map.encode_key(OpRefType::Delete.path().to_string())?;
                map.encode_value((path, key))?
            }
        }

        map.end()
    }
}

impl<'en> IntoStream<'en> for OpRef {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        let mut map = e.encode_map(Some(1))?;

        match self {
            OpRef::Get((path, key)) => map.encode_entry(path.to_string(), key)?,
            OpRef::Put((path, key, value)) => map.encode_entry(path.to_string(), (key, value))?,
            OpRef::Post((path, data)) => map.encode_entry(path.to_string(), data.into_inner())?,
            OpRef::Delete((path, key)) => {
                map.encode_key(OpRefType::Delete.path().to_string())?;
                map.encode_value((path, key))?
            }
        }

        map.end()
    }
}

impl fmt::Display for OpRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let class = self.class();

        match self {
            OpRef::Get((link, id)) => write!(f, "{} {}?key={}", class, link, id),
            OpRef::Put((path, id, val)) => write!(f, "{} {}?key={} <- {}", class, path, id, val),
            OpRef::Post((path, _)) => write!(f, "{} {}", class, path),
            OpRef::Delete((link, id)) => write!(f, "{} {}?key={}", class, link, id),
        }
    }
}
