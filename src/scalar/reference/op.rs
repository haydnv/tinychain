use std::collections::{HashMap, HashSet};
use std::fmt;

use async_trait::async_trait;
use futures::{try_join, TryFutureExt};
use log::debug;
use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCType};
use crate::error;
use crate::general::{Map, TCResult};
use crate::handler::Public;
use crate::request::Request;
use crate::scalar::{
    label, Id, Link, PathSegment, Scalar, ScalarClass, ScalarInstance, TCPathBuf, TryCastFrom,
    TryCastInto, Value,
};
use crate::transaction::Txn;

use super::{IdRef, RefType, Refer, TCRef};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum MethodType {
    Get,
    Put,
    Post,
    Delete,
}

impl Class for MethodType {
    type Instance = Method;
}

impl NativeClass for MethodType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "get" => Ok(Self::Get),
                "put" => Ok(Self::Put),
                "post" => Ok(Self::Post),
                "delete" => Ok(Self::Delete),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        RefType::prefix().append(label("method"))
    }
}

impl ScalarClass for MethodType {
    type Instance = Method;

    fn try_cast<S>(&self, scalar: S) -> TCResult<Method>
    where
        Scalar: From<S>,
    {
        let scalar = Scalar::from(scalar);

        let method = match self {
            Self::Get => {
                let method =
                    scalar.try_cast_into(|s| error::bad_request("Invalid GET method", s))?;
                Method::Get(method)
            }
            Self::Put => {
                let method =
                    scalar.try_cast_into(|s| error::bad_request("Invalid PUT method", s))?;
                Method::Put(method)
            }
            Self::Post => {
                let method =
                    scalar.try_cast_into(|s| error::bad_request("Invalid POST method", s))?;
                Method::Post(method)
            }
            Self::Delete => {
                let method =
                    scalar.try_cast_into(|s| error::bad_request("Invalid DELETE method", s))?;
                Method::Delete(method)
            }
        };

        Ok(method)
    }
}

impl From<MethodType> for Link {
    fn from(mt: MethodType) -> Link {
        use MethodType::*;
        let suffix = match mt {
            Get => label("get"),
            Put => label("put"),
            Post => label("post"),
            Delete => label("delete"),
        };

        MethodType::prefix().append(suffix).into()
    }
}

impl From<MethodType> for TCType {
    fn from(mt: MethodType) -> TCType {
        RefType::Method(mt).into()
    }
}

impl fmt::Display for MethodType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET method"),
            Self::Put => write!(f, "type: PUT method"),
            Self::Post => write!(f, "type: POST method"),
            Self::Delete => write!(f, "type: DELETE method"),
        }
    }
}

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
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "get" => Ok(OpRefType::Get),
                "put" => Ok(OpRefType::Put),
                "post" => Ok(OpRefType::Post),
                "delete" => Ok(OpRefType::Delete),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        RefType::prefix().append(label("op"))
    }
}

impl ScalarClass for OpRefType {
    type Instance = OpRef;

    fn try_cast<S>(&self, scalar: S) -> TCResult<OpRef>
    where
        Scalar: From<S>,
    {
        Err(error::bad_request(
            format!("Cannot cast into {} from", self),
            Scalar::from(scalar),
        ))
    }
}

impl From<OpRefType> for Link {
    fn from(ort: OpRefType) -> Link {
        use OpRefType as ORT;
        let suffix = match ort {
            ORT::Get => label("get"),
            ORT::Put => label("put"),
            ORT::Post => label("post"),
            ORT::Delete => label("delete"),
        };

        ORT::prefix().append(suffix).into()
    }
}

impl From<OpRefType> for TCType {
    fn from(ort: OpRefType) -> TCType {
        RefType::Op(ort).into()
    }
}

impl fmt::Display for OpRefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET Op ref"),
            Self::Put => write!(f, "type: PUT Op ref"),
            Self::Post => write!(f, "type: POST Op ref"),
            Self::Delete => write!(f, "type: DELETE Op ref"),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum Key {
    Ref(IdRef),
    Value(Value),
}

impl Key {
    pub async fn resolve_value(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<Value> {
        const ERR: &str = "Key must be a Value, not";

        let key = self.resolve(request, txn, context).await?;
        match key {
            State::Scalar(scalar) => scalar.try_cast_into(|s| error::bad_request(ERR, s)),
            other => Err(error::bad_request(ERR, other)),
        }
    }
}

#[async_trait]
impl Refer for Key {
    fn requires(&self, deps: &mut HashSet<Id>) {
        if let Self::Ref(id_ref) = self {
            deps.insert(id_ref.id().clone());
        }
    }

    async fn resolve(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<State> {
        match self {
            Self::Ref(id_ref) => id_ref.resolve(request, txn, context).await,
            Self::Value(value) => Ok(State::from(value)),
        }
    }
}

impl TryCastFrom<Scalar> for Key {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => match &**tc_ref {
                TCRef::Id(_) => true,
                _ => false,
            },
            Scalar::Value(_) => true,
            Scalar::Tuple(tuple) => Value::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Key> {
        match scalar {
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Some(Key::Ref(id_ref)),
                _ => None,
            },
            Scalar::Value(value) => Some(Key::Value(value)),
            Scalar::Tuple(tuple) => Value::opt_cast_from(tuple).map(Key::Value),
            _ => None,
        }
    }
}

impl From<Key> for Scalar {
    fn from(key: Key) -> Scalar {
        match key {
            Key::Ref(id_ref) => Scalar::Ref(Box::new(TCRef::Id(id_ref))),
            Key::Value(value) => Scalar::Value(value),
        }
    }
}

impl Serialize for Key {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Key::Ref(id_ref) => id_ref.serialize(s),
            Key::Value(value) => value.serialize(s),
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

type GetMethod = (IdRef, TCPathBuf, Key);
type PutMethod = (IdRef, TCPathBuf, Key, Scalar);
type PostMethod = (IdRef, TCPathBuf, Map<Scalar>);
type DeleteMethod = (IdRef, TCPathBuf, Key);

#[derive(Clone, Eq, PartialEq)]
pub enum Method {
    Get(GetMethod),
    Put(PutMethod),
    Post(PostMethod),
    Delete(DeleteMethod),
}

impl Instance for Method {
    type Class = MethodType;

    fn class(&self) -> MethodType {
        match self {
            Self::Get(_) => MethodType::Get,
            Self::Put(_) => MethodType::Put,
            Self::Post(_) => MethodType::Post,
            Self::Delete(_) => MethodType::Delete,
        }
    }
}

impl ScalarInstance for Method {
    type Class = MethodType;
}

#[async_trait]
impl Refer for Method {
    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Method::Get((subject, _path, key)) => {
                deps.insert(subject.id().clone());

                if let Key::Ref(id_ref) = key {
                    deps.insert(id_ref.id().clone());
                }
            }
            Method::Put((subject, _path, key, value)) => {
                deps.insert(subject.id().clone());

                if let Key::Ref(key) = key {
                    deps.insert(key.id().clone());
                }

                value.requires(deps);
            }
            Method::Post((subject, _path, _params)) => {
                deps.insert(subject.id().clone());
            }
            Method::Delete((subject, _path, key)) => {
                deps.insert(subject.id().clone());

                if let Key::Ref(tc_ref) = key {
                    deps.insert(tc_ref.id().clone());
                }
            }
        }
    }

    async fn resolve(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<State> {
        match self {
            Self::Get((subject, path, key)) => {
                let (subject, key) = try_join!(
                    subject.resolve(request, txn, context),
                    key.resolve_value(request, txn, context)
                )?;

                subject.get(request, txn, &path, key).await
            }
            Self::Put((subject, path, key, value)) => {
                let (subject, key, value) = try_join!(
                    subject.resolve(request, txn, context),
                    key.resolve_value(request, txn, context),
                    value.resolve(request, txn, context)
                )?;

                subject
                    .put(request, txn, &path, key, value)
                    .map_ok(State::from)
                    .await
            }
            Self::Post((subject, path, params)) => {
                let (subject, params) = try_join!(
                    subject.resolve(request, txn, context),
                    params.resolve(request, txn, context)
                )?;

                debug!("Method::resolve {}{}: {})", subject, path, params);

                // TODO: update Public::post to accept a Map<State>
                if let State::Scalar(Scalar::Map(params)) = params {
                    subject.post(request, txn, &path, params).await
                } else {
                    Err(error::not_implemented(format!(
                        "POST with params {}",
                        params
                    )))
                }
            }
            Self::Delete((subject, path, key)) => {
                let (subject, key) = try_join!(
                    subject.resolve(request, txn, context),
                    key.resolve_value(request, txn, context)
                )?;

                subject
                    .delete(request, txn, &path, key)
                    .map_ok(State::from)
                    .await
            }
        }
    }
}

impl Serialize for Method {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(1))?;

        match self {
            Method::Get((subject, path, key)) => {
                map.serialize_entry(&format!("{}{}", subject, path), &(key,))?
            }
            Method::Put((subject, path, key, value)) => {
                map.serialize_entry(&format!("{}{}", subject, path), &(key, value))?
            }
            Method::Post((subject, path, args)) => {
                map.serialize_entry(&format!("{}{}", subject, path), args)?
            }
            Method::Delete((subject, path, key)) => {
                map.serialize_key(&Link::from(MethodType::Delete).path().to_string())?;
                map.serialize_value(&(subject, path, key))?
            }
        };

        map.end()
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get((subject, path, key)) => write!(f, "GET {}{}?key={}", subject, path, key),
            Self::Put((subject, path, key, value)) => {
                write!(f, "PUT {}{}?key={} <- {}", subject, path, key, value)
            }
            Self::Post((subject, path, params)) => {
                write!(f, "POST {}{} args: {}", subject, path, params)
            }
            Self::Delete((subject, path, key)) => {
                write!(f, "DELETE {}{}?key={}", subject, path, key)
            }
        }
    }
}

type GetRef = (Link, Key);
type PutRef = (Link, Key, Scalar);
type PostRef = (Link, Map<Scalar>);
type DeleteRef = (Link, Key);

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
        match self {
            Self::Get(_) => OpRefType::Get,
            Self::Put(_) => OpRefType::Put,
            Self::Post(_) => OpRefType::Post,
            Self::Delete(_) => OpRefType::Delete,
        }
    }
}

impl ScalarInstance for OpRef {
    type Class = OpRefType;
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

                value.requires(deps);
            }
            OpRef::Post((_path, params)) => {
                for provider in params.values() {
                    provider.requires(deps);
                }
            }
            OpRef::Delete((_path, key)) => {
                if let Key::Ref(id_ref) = key {
                    deps.insert(id_ref.id().clone());
                }
            }
        }
    }

    async fn resolve(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<State> {
        txn.resolve_op(request, context, self).await
    }
}

impl Serialize for OpRef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(1))?;

        match self {
            OpRef::Get((path, key)) => map.serialize_entry(&path.to_string(), key)?,
            OpRef::Put((path, key, value)) => {
                map.serialize_entry(&path.to_string(), &(key, value))?
            }
            OpRef::Post((path, data)) => map.serialize_entry(&path.to_string(), data)?,
            OpRef::Delete((path, key)) => {
                map.serialize_key(&Link::from(OpRefType::Delete).path().to_string())?;
                map.serialize_value(&(path, key))?
            }
        }

        map.end()
    }
}

impl fmt::Display for OpRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpRef::Get((link, id)) => write!(f, "OpRef::Get {}?key={}", link, id),
            OpRef::Put((path, id, val)) => write!(f, "OpRef::Put {}?key={} <- {}", path, id, val),
            OpRef::Post((path, _)) => write!(f, "OpRef::Post {}", path),
            OpRef::Delete((link, id)) => write!(f, "OpRef::Delete {}?key={}", link, id),
        }
    }
}
