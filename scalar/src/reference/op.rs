use std::collections::HashSet;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use destream::en::{EncodeMap, Encoder, IntoStream, ToStream};

use generic::*;

use crate::{Link, Scalar, Value};

use super::{IdRef, RefInstance, TCRef};

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
pub enum Key {
    Ref(IdRef),
    Value(Value),
}

#[async_trait]
impl RefInstance for Key {
    fn requires(&self, deps: &mut HashSet<Id>) {
        if let Self::Ref(id_ref) = self {
            deps.insert(id_ref.id().clone());
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
        use OpRefType as ORT;
        match self {
            Self::Get(_) => ORT::Get,
            Self::Put(_) => ORT::Put,
            Self::Post(_) => ORT::Post,
            Self::Delete(_) => ORT::Delete,
        }
    }
}

impl RefInstance for OpRef {
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
