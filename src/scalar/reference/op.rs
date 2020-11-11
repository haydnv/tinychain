use std::fmt;

use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::{Class, Instance, NativeClass, TCResult};
use crate::error;
use crate::scalar::{
    label, Link, Object, PathSegment, Scalar, ScalarClass, ScalarInstance, TCPathBuf, TryCastFrom,
    TryCastInto, Value,
};

use super::{IdRef, RefType, TCRef};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum MethodType {
    Get,
    Put,
    Post,
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

    fn try_cast<S: Into<Scalar>>(&self, _scalar: S) -> TCResult<Method> {
        Err(error::not_implemented("Cast Scalar into Method"))
    }
}

impl From<MethodType> for Link {
    fn from(mt: MethodType) -> Link {
        let suffix = match mt {
            MethodType::Get => label("get"),
            MethodType::Put => label("put"),
            MethodType::Post => label("post"),
        };

        MethodType::prefix().append(suffix).into()
    }
}

impl fmt::Display for MethodType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET method"),
            Self::Put => write!(f, "type: PUT method"),
            Self::Post => write!(f, "type: POST method"),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpRefType {
    Get,
    Put,
    Post,
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

    fn try_cast<S: Into<Scalar>>(&self, scalar: S) -> TCResult<OpRef> {
        let scalar: Scalar = scalar.into();
        scalar.try_cast_into(|v| error::bad_request(format!("Cannot cast into {} from", self), v))
    }
}

impl From<OpRefType> for Link {
    fn from(ort: OpRefType) -> Link {
        use OpRefType as ORT;
        let suffix = match ort {
            ORT::Get => label("get"),
            ORT::Put => label("put"),
            ORT::Post => label("post"),
        };

        ORT::prefix().append(suffix).into()
    }
}

impl fmt::Display for OpRefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "type: GET Op ref"),
            Self::Put => write!(f, "type: PUT Op ref"),
            Self::Post => write!(f, "type: POST Op ref"),
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
            Scalar::Ref(_) => true,
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

#[derive(Clone, Eq, PartialEq)]
pub enum Method {
    Get((IdRef, TCPathBuf), Key),
    Put((IdRef, TCPathBuf), (Key, Scalar)),
    Post((IdRef, TCPathBuf), Object),
}

impl Instance for Method {
    type Class = MethodType;

    fn class(&self) -> MethodType {
        match self {
            Self::Get(_, _) => MethodType::Get,
            Self::Put(_, _) => MethodType::Put,
            Self::Post(_, _) => MethodType::Post,
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
            Method::Get((subject, path), arg) => {
                map.serialize_entry(&format!("{}{}", subject, path), &(arg,))
            }
            Method::Put((subject, path), args) => {
                map.serialize_entry(&format!("{}{}", subject, path), args)
            }
            Method::Post((subject, path), args) => {
                map.serialize_entry(&format!("{}{}", subject, path), args)
            }
        }?;

        map.end()
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get((subject, path), key) => write!(f, "GET {}: {}?key={}", subject, path, key),
            Self::Put((subject, path), (key, value)) => {
                write!(f, "PUT {}{}?key={} <- {}", subject, path, key, value)
            }
            Self::Post((subject, path), params) => {
                write!(f, "POST {}{} args: {}", subject, path, params)
            }
        }
    }
}

type GetRef = (Link, Key);
type PutRef = (Link, Key, Scalar);
type PostRef = (Link, Object);

#[derive(Clone, Eq, PartialEq)]
pub enum OpRef {
    Get(GetRef),
    Put(PutRef),
    Post(PostRef),
}

impl Instance for OpRef {
    type Class = OpRefType;

    fn class(&self) -> OpRefType {
        match self {
            Self::Get(_) => OpRefType::Get,
            Self::Put(_) => OpRefType::Put,
            Self::Post(_) => OpRefType::Post,
        }
    }
}

impl ScalarInstance for OpRef {
    type Class = OpRefType;
}

impl TryCastFrom<Scalar> for OpRef {
    fn can_cast_from(s: &Scalar) -> bool {
        s.matches::<PostRef>() || s.matches::<PutRef>() || s.matches::<GetRef>()
    }

    fn opt_cast_from(_s: Scalar) -> Option<OpRef> {
        unimplemented!()
    }
}

impl Serialize for OpRef {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            OpRef::Get((path, key)) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(&path.to_string(), key)?;
                map.end()
            }
            OpRef::Put((path, key, value)) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(&path.to_string(), &(key, value))?;
                map.end()
            }
            OpRef::Post((path, data)) => {
                let mut map = s.serialize_map(Some(1))?;
                map.serialize_entry(&path.to_string(), data)?;
                map.end()
            }
        }
    }
}

impl fmt::Display for OpRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpRef::Get((link, id)) => write!(f, "OpRef::Get {}: {}", link, id),
            OpRef::Put((path, id, val)) => write!(f, "OpRef::Put {}: {} <- {}", path, id, val),
            OpRef::Post((path, _)) => write!(f, "OpRef::Post {}", path),
        }
    }
}
