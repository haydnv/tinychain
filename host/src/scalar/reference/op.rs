//! Resolve a reference to an op.

use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::{self, Decoder, FromStream};
use destream::en::{EncodeMap, Encoder, IntoStream, ToStream};
use futures::{try_join, TryFutureExt};
use log::debug;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::*;

use crate::route::Public;
use crate::scalar::{Link, Scalar, Scope, Value, SELF};
use crate::state::State;
use crate::txn::Txn;

use super::{IdRef, Refer, TCRef};

const PREFIX: PathLabel = path_label(&["state", "scalar", "ref", "op"]);

/// The [`Class`] of an [`OpRef`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OpRefType {
    Get,
    Put,
    Post,
    Delete,
}

impl Class for OpRefType {}

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

/// The subject of an op.
#[derive(Clone, Eq, PartialEq)]
pub enum Subject {
    Link(Link),
    Ref(IdRef, TCPathBuf),
}

impl Subject {
    fn is_view(&self) -> bool {
        if let Self::Ref(id, _path) = self {
            id.is_view()
        } else {
            false
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Ref(id_ref, _) if id_ref.id() != &SELF => id_ref.requires(deps),
            _ => {}
        }
    }
}

impl FromStr for Subject {
    type Err = TCError;

    fn from_str(s: &str) -> TCResult<Self> {
        if s.starts_with('$') {
            if let Some(i) = s.find('/') {
                let id_ref = IdRef::from_str(&s[..i])?;
                let path = TCPathBuf::from_str(&s[i..])?;
                Ok(Self::Ref(id_ref, path))
            } else {
                let id_ref = IdRef::from_str(s)?;
                Ok(Self::Ref(id_ref, TCPathBuf::default()))
            }
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
            Self::Ref(id_ref, path) if path.is_empty() => id_ref.to_stream(e),
            Self::Ref(id_ref, path) => format!("{}{}", id_ref, path).into_stream(e),
        }
    }
}

impl<'en> IntoStream<'en> for Subject {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Link(link) => link.into_stream(e),
            Self::Ref(id_ref, path) if path.is_empty() => id_ref.into_stream(e),
            Self::Ref(id_ref, path) => format!("{}{}", id_ref, path).into_stream(e),
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
                    Some(Self::Ref(id_ref, TCPathBuf::default()))
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
                TCRef::Id(id_ref) => Some(Self::Ref(id_ref, TCPathBuf::default())),
                _ => None,
            },
            Scalar::Value(value) => Self::opt_cast_from(value),
            _ => None,
        }
    }
}

impl From<TCPathBuf> for Subject {
    fn from(path: TCPathBuf) -> Self {
        Self::Link(path.into())
    }
}

impl From<Link> for Subject {
    fn from(link: Link) -> Self {
        Self::Link(link)
    }
}

impl From<(IdRef, TCPathBuf)> for Subject {
    fn from(get: (IdRef, TCPathBuf)) -> Self {
        Self::Ref(get.0, get.1)
    }
}

impl TryFrom<Subject> for Link {
    type Error = TCError;

    fn try_from(subject: Subject) -> TCResult<Self> {
        match subject {
            Subject::Link(link) => Ok(link),
            other => Err(TCError::bad_request("expected a Link but found", other)),
        }
    }
}

impl TryFrom<Subject> for TCPathBuf {
    type Error = TCError;

    fn try_from(subject: Subject) -> TCResult<Self> {
        match subject {
            Subject::Link(link) if link.host().is_none() => Ok(link.into_path()),
            other => Err(TCError::bad_request("expected a Path but found", other)),
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Link(link) => fmt::Display::fmt(link, f),
            Self::Ref(id_ref, path) if path.is_empty() => fmt::Display::fmt(id_ref, f),
            Self::Ref(id_ref, path) => write!(f, "{}{}", id_ref, path),
        }
    }
}

/// The data defining a reference to a GET op.
pub type GetRef = (Subject, Scalar);

/// The data defining a reference to a PUT op.
pub type PutRef = (Subject, Scalar, Scalar);

/// The data defining a reference to a POST op.
pub type PostRef = (Subject, Map<Scalar>);

/// The data defining a reference to a DELETE op.
pub type DeleteRef = (Subject, Scalar);

/// A reference to an op.
#[derive(Clone, Eq, PartialEq)]
pub enum OpRef {
    Get(GetRef),
    Put(PutRef),
    Post(PostRef),
    Delete(DeleteRef),
}

impl OpRef {
    fn subject(&self) -> &Subject {
        match self {
            Self::Get((subject, _)) => subject,
            Self::Put((subject, _, _)) => subject,
            Self::Post((subject, _)) => subject,
            Self::Delete((subject, _)) => subject,
        }
    }
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
    fn is_view(&self) -> bool {
        self.subject().is_view()
    }

    fn is_write(&self) -> bool {
        match self {
            Self::Get(_) => false,
            Self::Put(_) => true,
            Self::Post(_) => false,
            Self::Delete(_) => true,
        }
    }

    fn is_derived_write(&self) -> bool {
        self.is_view() && self.is_write()
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Get((subject, key)) => {
                subject.requires(deps);
                key.requires(deps);
            }
            Self::Put((subject, key, value)) => {
                subject.requires(deps);
                key.requires(deps);
                value.requires(deps);
            }
            Self::Post((subject, params)) => {
                subject.requires(deps);
                for param in params.values() {
                    param.requires(deps);
                }
            }
            Self::Delete((subject, key)) => {
                subject.requires(deps);
                key.requires(deps);
            }
        }
    }

    async fn resolve<'a, T: Instance + Public>(
        self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        debug!("OpRef::resolve {} from context {:?}", self, context);

        let invalid_key = |v: &State| TCError::bad_request("invalid key", v);

        match self {
            Self::Get((subject, key)) => match subject {
                Subject::Link(link) => {
                    let key = key.resolve(context, txn).await?;
                    let key = key.try_cast_into(invalid_key)?;

                    txn.get(link, key).await
                }
                Subject::Ref(id_ref, path) => {
                    let key = key.resolve(context, txn).await?;
                    let key = key.try_cast_into(invalid_key)?;

                    context.resolve_get(txn, id_ref.id(), &path, key).await
                }
            },
            Self::Put((subject, key, value)) => match subject {
                Subject::Link(link) => {
                    let key = key.resolve(context, txn).await?;
                    let key = key.try_cast_into(invalid_key)?;

                    let value = value.resolve(context, txn).await?;

                    txn.put(link, key, value).map_ok(State::from).await
                }
                Subject::Ref(id_ref, path) => {
                    let key = key.resolve(context, txn);
                    let value = value.resolve(context, txn);
                    let (key, value) = try_join!(key, value)?;

                    let key = key.try_cast_into(invalid_key)?;

                    context
                        .resolve_put(txn, id_ref.id(), &path, key, value)
                        .map_ok(State::from)
                        .await
                }
            },
            Self::Post((subject, params)) => match subject {
                Subject::Link(link) => {
                    let params = Scalar::Map(params).resolve(context, txn).await?;
                    txn.post(link, params).await
                }
                Subject::Ref(id_ref, path) => {
                    let params = Scalar::Map(params).resolve(context, txn).await?;
                    let params = params.try_into()?;

                    context.resolve_post(txn, id_ref.id(), &path, params).await
                }
            },
            Self::Delete((subject, key)) => match subject {
                Subject::Link(link) => {
                    let key = key.resolve(context, txn).await?;
                    let key = key.try_cast_into(invalid_key)?;

                    txn.delete(link, key).map_ok(State::from).await
                }
                Subject::Ref(id_ref, path) => {
                    let key = key.resolve(context, txn).await?;
                    let key = key.try_cast_into(invalid_key)?;

                    context
                        .resolve_delete(txn, id_ref.id(), &path, key)
                        .map_ok(State::from)
                        .await
                }
            },
        }
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
            Scalar::Tuple(params) if params.matches::<(Scalar, Scalar)>() => {
                let (key, value) = params.opt_cast_into().unwrap();
                Ok(OpRef::Put((subject, key, value)))
            }
            Scalar::Tuple(params) if params.matches::<(Scalar,)>() => {
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
            OpRef::Get((path, key)) => map.encode_entry(path.to_string(), (key,))?,
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
