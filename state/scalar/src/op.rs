use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::{Decoder, Error, FromStream, MapAccess, Visitor};
use destream::en::{EncodeMap, Encoder, ToStream};
use safecast::{Match, TryCastFrom, TryCastInto};

use generic::*;

use super::Scalar;

const PREFIX: PathLabel = path_label(&["state", "scalar", "op"]);

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum OpDefType {
    Get,
    Put,
    Post,
    Delete,
}

impl Class for OpDefType {
    type Instance = OpDef;
}

impl NativeClass for OpDefType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 4 && &path[..3] == &PREFIX[..] {
            match path[3].as_str() {
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
        let prefix = TCPathBuf::from(PREFIX);

        let suffix = match self {
            Self::Get => "get",
            Self::Put => "put",
            Self::Post => "post",
            Self::Delete => "delete",
        };

        prefix.append(label(suffix)).into()
    }
}

impl fmt::Display for OpDefType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get => write!(f, "GET Op definition"),
            Self::Put => write!(f, "PUT Op definition"),
            Self::Post => write!(f, "POST Op definition"),
            Self::Delete => write!(f, "DELETE Op definition"),
        }
    }
}

pub type GetOp = (Id, Vec<(Id, Scalar)>);
pub type PutOp = (Id, Id, Vec<(Id, Scalar)>);
pub type PostOp = Vec<(Id, Scalar)>;
pub type DeleteOp = (Id, Vec<(Id, Scalar)>);

#[derive(Clone, Eq, PartialEq)]
pub enum OpDef {
    Get(GetOp),
    Put(PutOp),
    Post(PostOp),
    Delete(DeleteOp),
}

impl Instance for OpDef {
    type Class = OpDefType;

    fn class(&self) -> OpDefType {
        match self {
            Self::Get(_) => OpDefType::Get,
            Self::Put(_) => OpDefType::Put,
            Self::Post(_) => OpDefType::Post,
            Self::Delete(_) => OpDefType::Delete,
        }
    }
}

impl TryCastFrom<Scalar> for OpDef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<PutOp>() || scalar.matches::<GetOp>() || scalar.matches::<PostOp>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<OpDef> {
        if scalar.matches::<PutOp>() {
            scalar.opt_cast_into().map(OpDef::Put)
        } else if scalar.matches::<GetOp>() {
            scalar.opt_cast_into().map(OpDef::Get)
        } else if scalar.matches::<PostOp>() {
            scalar.opt_cast_into().map(OpDef::Post)
        } else {
            None
        }
    }
}

pub struct OpDefVisitor;

impl OpDefVisitor {
    pub async fn visit_map_value<A: MapAccess>(
        class: OpDefType,
        mut map: A,
    ) -> Result<OpDef, A::Error> {
        use OpDefType as ODT;

        match class {
            ODT::Get => {
                let op = map.next_value().await?;
                Ok(OpDef::Get(op))
            }
            ODT::Put => {
                let op = map.next_value().await?;
                Ok(OpDef::Put(op))
            }
            ODT::Post => {
                let op = map.next_value().await?;
                Ok(OpDef::Post(op))
            }
            ODT::Delete => {
                let op = map.next_value().await?;
                Ok(OpDef::Delete(op))
            }
        }
    }
}

#[async_trait]
impl Visitor for OpDefVisitor {
    type Value = OpDef;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("an Op definition")
    }

    async fn visit_map<A: MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let err = || A::Error::custom("Expected an Op definition type, e.g. \"/state/op/get\"");

        let class = map.next_key::<String>().await?.ok_or_else(err)?;
        let class = TCPathBuf::from_str(&class).map_err(A::Error::custom)?;
        let class = OpDefType::from_path(&class).ok_or_else(err)?;

        Self::visit_map_value(class, map).await
    }
}

#[async_trait]
impl FromStream for OpDef {
    async fn from_stream<D: Decoder>(decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(OpDefVisitor).await
    }
}

impl<'en> ToStream<'en> for OpDef {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        let class = self.class().to_string();
        let mut map = e.encode_map(Some(1))?;

        match self {
            Self::Get(def) => map.encode_entry(class, def),
            Self::Put(def) => map.encode_entry(class, def),
            Self::Post(def) => map.encode_entry(class, def),
            Self::Delete(def) => map.encode_entry(class, def),
        }?;

        map.end()
    }
}

impl fmt::Display for OpDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Get(_) => write!(f, "GET Op"),
            Self::Put(_) => write!(f, "PUT Op"),
            Self::Post(_) => write!(f, "POST Op"),
            Self::Delete(_) => write!(f, "DELETE Op"),
        }
    }
}
