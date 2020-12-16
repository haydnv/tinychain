use std::fmt;

use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::{Class, Instance, NativeClass, TCResult, TCType};
use crate::error;
use crate::scalar::{
    label, Link, Map, PathSegment, Scalar, ScalarClass, ScalarInstance, TCPathBuf, TryCastFrom,
    TryCastInto, Value,
};

use super::{IdRef, RefType, TCRef};

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
type PostMethod = (IdRef, TCPathBuf, Map);
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
type PostRef = (Link, Map);
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
