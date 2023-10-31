//! Resolve a reference to an op.

use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::Deref;
use std::str::FromStr;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use futures::{try_join, TryFutureExt};
use get_size::GetSize;
use get_size_derive::*;
use log::debug;
use safecast::{CastFrom, CastInto, Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::public::{Public, StateInstance, ToState};
use tc_transact::RPCClient;
use tc_value::Value;
use tcgeneric::*;

use crate::{Link, Scalar, Scope, SELF};

use super::{IdRef, OpDef, Refer, TCRef};

const PREFIX: PathLabel = path_label(&["state", "scalar", "ref", "op"]);

/// The [`Class`] of an [`OpRef`].
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
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

impl fmt::Debug for OpRefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "GET"),
            Self::Put => write!(f, "PUT"),
            Self::Post => write!(f, "POST"),
            Self::Delete => write!(f, "DELETE"),
        }
    }
}

/// The subject of an op.
#[derive(Clone, Eq, PartialEq, GetSize)]
pub enum Subject {
    Link(Link),
    Ref(IdRef, TCPathBuf),
}

impl Subject {
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Ref(id_ref, suffix) if id_ref.id() == &SELF => {
                let mut path = path.clone();
                path.extend(suffix);
                Self::Link(path.into())
            }
            other => other,
        }
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Link(link) if link.path().starts_with(path) => Self::Ref(
                IdRef::from(SELF),
                link.into_path().into_iter().skip(path.len()).collect(),
            ),
            other => other,
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Ref(id_ref, _) if id_ref.id() != &SELF => {
                deps.insert(id_ref.id().clone());
            }
            _ => {}
        }
    }
}

impl<'a, D: Digest> Hash<D> for &'a Subject {
    fn hash(self) -> Output<D> {
        match self {
            Subject::Link(link) => Hash::<D>::hash(link),
            Subject::Ref(id, path) => Hash::<D>::hash((id, path)),
        }
    }
}

impl From<IdRef> for Subject {
    fn from(id_ref: IdRef) -> Self {
        Self::Ref(id_ref, TCPathBuf::default())
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
            Link::from_str(s).map(Self::Link).map_err(TCError::from)
        }
    }
}

impl From<Subject> for Scalar {
    fn from(subject: Subject) -> Self {
        match subject {
            Subject::Ref(id_ref, path) => Scalar::Tuple(Tuple::from((id_ref.into(), path.into()))),
            Subject::Link(link) => Scalar::Value(link.into()),
        }
    }
}

#[async_trait]
impl de::FromStream for Subject {
    type Context = ();

    async fn from_stream<D: de::Decoder>(context: (), d: &mut D) -> Result<Self, D::Error> {
        let subject = String::from_stream(context, d).await?;
        Subject::from_str(&subject).map_err(de::Error::custom)
    }
}

impl<'en> en::ToStream<'en> for Subject {
    fn to_stream<E: en::Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Link(link) => link.to_stream(e),
            Self::Ref(id_ref, path) if path.is_empty() => id_ref.to_stream(e),
            Self::Ref(id_ref, path) => {
                en::IntoStream::into_stream(format!("{}{}", id_ref, path), e)
            }
        }
    }
}

impl<'en> en::IntoStream<'en> for Subject {
    fn into_stream<E: en::Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
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
            Value::String(s) => Self::from_str(s.as_str()).is_ok(),
            Value::Tuple(tuple) => tuple.matches::<(IdRef, TCPathBuf)>(),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Link(link) => Some(Self::Link(link)),
            Value::String(s) => Self::from_str(s.as_str()).ok(),
            Value::Tuple(tuple) => tuple
                .opt_cast_into()
                .map(|(id, path)| Self::from((id, path))),
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
            Scalar::Tuple(tuple) => tuple.matches::<(IdRef, TCPathBuf)>(),
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
            Scalar::Tuple(tuple) => tuple
                .opt_cast_into()
                .map(|(id, path)| Self::from((id, path))),

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
            other => Err(TCError::unexpected(other, "a Link")),
        }
    }
}

impl TryFrom<Subject> for TCPathBuf {
    type Error = TCError;

    fn try_from(subject: Subject) -> TCResult<Self> {
        match subject {
            Subject::Link(link) if link.host().is_none() => Ok(link.into_path()),
            other => Err(TCError::unexpected(other, "a Path")),
        }
    }
}

impl fmt::Debug for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Link(link) => fmt::Debug::fmt(link, f),
            Self::Ref(id_ref, path) if path.is_empty() => fmt::Debug::fmt(id_ref, f),
            Self::Ref(id_ref, path) => write!(f, "subject: {:?} {:?}", id_ref, path),
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
#[derive(Clone, Eq, PartialEq, GetSize)]
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

impl OpRef {
    pub(crate) fn dereference_subject(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Get((subject, key)) => Self::Get((subject.dereference_self(path), key)),
            Self::Put((subject, key, value)) => {
                Self::Put((subject.dereference_self(path), key, value))
            }
            Self::Post((subject, params)) => Self::Post((subject.dereference_self(path), params)),
            Self::Delete((subject, key)) => Self::Delete((subject.dereference_self(path), key)),
        }
    }
}

#[async_trait]
impl<State> Refer<State> for OpRef
where
    State: StateInstance + Refer<State> + From<Scalar>,
    State::Closure: From<(Map<State>, OpDef)> + TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
    Value: TryFrom<State, Error = TCError> + TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Get((subject, key)) => Self::Get((subject.dereference_self(path), key)),
            Self::Put((subject, key, value)) => Self::Put((
                subject.dereference_self(path),
                key,
                value.dereference_self(path),
            )),
            Self::Post((subject, params)) => {
                if let Scalar::Map(params) = Scalar::Map(params).dereference_self(path) {
                    Self::Post((subject.dereference_self(path), params))
                } else {
                    panic!("Scalar::Map::dereference_self did not return a Scalar::Map")
                }
            }
            Self::Delete((subject, key)) => Self::Delete((subject.dereference_self(path), key)),
        }
    }

    fn is_conditional(&self) -> bool {
        match self {
            Self::Get((_, key)) => key.is_conditional(),
            Self::Put((_, key, value)) => key.is_conditional() || value.is_conditional(),
            Self::Post((_, params)) => params.values().any(|scalar| scalar.is_conditional()),
            Self::Delete((_, key)) => key.is_conditional(),
        }
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        let subject = match self {
            Self::Put((Subject::Link(link), _, _)) => Some(link),
            Self::Delete((Subject::Link(link), _)) => Some(link),
            _ => None,
        };

        if let Some(link) = subject {
            !link.path().starts_with(cluster_path)
        } else {
            false
        }
    }

    fn is_ref(&self) -> bool {
        true
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Get((subject, key)) => Self::Get((subject.reference_self(path), key)),
            Self::Put((subject, key, value)) => Self::Put((
                subject.reference_self(path),
                key,
                value.reference_self(path),
            )),
            Self::Post((subject, params)) => {
                let params = if let Scalar::Map(params) = Scalar::Map(params).reference_self(path) {
                    params
                } else {
                    panic!("Scalar::Map::reference_self did not return a Scalar::Map")
                };

                Self::Post((subject.reference_self(path), params))
            }
            Self::Delete((subject, key)) => Self::Delete((subject.reference_self(path), key)),
        }
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

    async fn resolve<'a, T: ToState<State> + Public<State> + Instance>(
        self,
        context: &'a Scope<'a, State, T>,
        txn: &'a State::Txn,
    ) -> TCResult<State> {
        debug!("OpRef::resolve {:?} from context {:?}", self, context);

        #[inline]
        fn invalid_key<'a, State: fmt::Debug, T>(
            subject: &'a T,
        ) -> impl FnOnce(&State) -> TCError + 'a
        where
            T: fmt::Debug + 'a,
        {
            move |v| bad_request!("{:?} is not a valid key for {:?}", v, subject)
        }

        match self {
            Self::Get((subject, key)) => match subject {
                Subject::Link(link) => {
                    let key = resolve(key, context, txn).await?;
                    let key = Value::try_cast_from(key, invalid_key(&link))?;
                    txn.get(link, key).await
                }
                Subject::Ref(id_ref, path) => {
                    let key = resolve(key, context, txn).await?;
                    let key = key.try_cast_into(invalid_key(&id_ref))?;
                    context.resolve_get(txn, id_ref.id(), &path, key).await
                }
            },
            Self::Put((subject, key, value)) => match subject {
                Subject::Link(link) => {
                    let key = resolve(key, context, txn).await?;
                    let key = Value::try_cast_from(key, invalid_key(&link))?;

                    let value = resolve(value, context, txn).await?;

                    txn.put(link, key, value)
                        .map_ok(|()| State::default())
                        .await
                }
                Subject::Ref(id_ref, path) => {
                    let key = resolve(key, context, txn);
                    let value = resolve(value, context, txn);
                    let (key, value) = try_join!(key, value)?;

                    let key = key.try_cast_into(invalid_key(&id_ref))?;

                    context
                        .resolve_put(txn, id_ref.id(), &path, key, value)
                        .map_ok(|()| State::default())
                        .await
                }
            },
            Self::Post((subject, params)) => match subject {
                Subject::Link(link) => {
                    let params = resolve(Scalar::Map(params), context, txn).await?;
                    txn.post(link, params).await
                }
                Subject::Ref(id_ref, path) => {
                    let params = resolve(Scalar::Map(params), context, txn).await?;
                    let params = params.try_into()?;
                    context.resolve_post(txn, id_ref.id(), &path, params).await
                }
            },
            Self::Delete((subject, key)) => match subject {
                Subject::Link(link) => {
                    let key = resolve(key, context, txn).await?;
                    let key = Value::try_cast_from(key, invalid_key(&link))?;
                    txn.delete(link, key).map_ok(|()| State::default()).await
                }
                Subject::Ref(id_ref, path) => {
                    let key = resolve(key, context, txn).await?;
                    let key = key.try_cast_into(invalid_key(&id_ref))?;

                    context
                        .resolve_delete(txn, id_ref.id(), &path, key)
                        .map_ok(|()| State::default())
                        .await
                }
            },
        }
    }
}

impl<'a, D: Digest> Hash<D> for &'a OpRef {
    fn hash(self) -> Output<D> {
        match self {
            OpRef::Get(get) => Hash::<D>::hash(get),
            OpRef::Put(put) => Hash::<D>::hash(put),
            OpRef::Post((subject, params)) => Hash::<D>::hash((subject, params.deref())),
            OpRef::Delete(delete) => Hash::<D>::hash(delete),
        }
    }
}

impl CastFrom<OpRef> for Tuple<Scalar> {
    fn cast_from(value: OpRef) -> Self {
        match value {
            OpRef::Get((subject, key)) => (subject.into(), key).into(),
            OpRef::Put((subject, key, value)) => (subject.into(), key, value).into(),
            OpRef::Post((subject, params)) => (subject.into(), params.into()).cast_into(),
            OpRef::Delete((subject, key)) => (subject.into(), key).cast_into(),
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
        debug!("OpRefVisitor::visit_ref_value {} {:?}", subject, params);

        match params {
            Scalar::Map(params) => Ok(OpRef::Post((subject, params))),
            Scalar::Tuple(mut tuple) if tuple.len() == 1 => {
                let key = tuple.pop().unwrap();
                Ok(OpRef::Get((subject, key)))
            }
            Scalar::Tuple(mut tuple) if tuple.len() == 2 => {
                if let Subject::Link(link) = &subject {
                    if link.host().is_none() && link.path().len() >= 2 {
                        // TODO: is there a better place to put this? or a better implementation?
                        let is_chain = &link.path()[..2] == &["state", "chain"];
                        let is_collection = &link.path()[..2] == &["state", "collection"];

                        if is_chain || is_collection {
                            return Err(E::custom(format!(
                                "a scalar is immutable and cannot contain a mutable type {}: {:?} (consider using a closure)",
                                link, tuple
                            )));
                        }
                    }
                }

                let value = tuple.pop().unwrap();
                let key = tuple.pop().unwrap();

                if subject == Subject::Link(OpRefType::Delete.path().into()) {
                    let subject = key.try_cast_into(|k| {
                        E::invalid_type(format!("{k:?}"), "a Link or Id reference")
                    })?;

                    Ok(OpRef::Delete((subject, value)))
                } else {
                    Ok(OpRef::Put((subject, key, value)))
                }
            }
            other => {
                debug!(
                    "invalid parameters for method call on {:?}: {:?}",
                    subject, other
                );

                Err(de::Error::invalid_value(
                    format!("{other:?}"),
                    "OpRef parameters",
                ))
            }
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
impl de::FromStream for OpRef {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), d: &mut D) -> Result<Self, D::Error> {
        d.decode_map(OpRefVisitor).await
    }
}

impl<'en> en::ToStream<'en> for OpRef {
    fn to_stream<E: en::Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

        let mut map = e.encode_map(Some(1))?;

        match self {
            OpRef::Get((path, key)) => map.encode_entry(path.to_string(), (key,))?,
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

impl<'en> en::IntoStream<'en> for OpRef {
    fn into_stream<E: en::Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;

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

impl fmt::Debug for OpRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let class = self.class();

        match self {
            OpRef::Get((link, key)) => write!(f, "{:?} {}?key={:?}", class, link, key),
            OpRef::Put((path, key, value)) => {
                write!(f, "{:?} {}?key={:?} <- {:?}", class, path, key, value)
            }
            OpRef::Post((path, params)) => write!(f, "{:?} {}({:?})", class, path, params),
            OpRef::Delete((link, key)) => write!(f, "{:?} {}?key={:?}", class, link, key),
        }
    }
}

async fn resolve<'a, State, T, S>(
    tc_ref: T,
    context: &'a Scope<'a, State, S>,
    txn: &'a State::Txn,
) -> TCResult<State>
where
    State: StateInstance + Refer<State>,
    T: Refer<State>,
    S: ToState<State> + Public<State> + Instance,
{
    let mut state = tc_ref.resolve(context, txn).await?;
    while state.is_ref() {
        state = state.resolve(context, txn).await?;
    }

    Ok(state)
}
