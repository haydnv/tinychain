//! User-defined [`OpDef`]s

use std::fmt;
use std::str::FromStr;

use async_trait::async_trait;
use destream::de::{Decoder, Error, FromStream, MapAccess, Visitor};
use destream::en::{EncodeMap, Encoder, IntoStream, ToStream};
use log::debug;

use tc_error::*;
use tcgeneric::*;

use crate::scalar::{Executor, Scalar};
use crate::state::State;
use crate::txn::Txn;

const PREFIX: PathLabel = path_label(&["state", "scalar", "op"]);

/// The [`Class`] of a user-defined [`OpDef`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
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
            log::debug!(
                "OpDefType::from_path {} (type {})",
                TCPath::from(path),
                &path[3]
            );

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

/// A GET handler.
pub type GetOp = (Id, Vec<(Id, Scalar)>);

/// A PUT handler.
pub type PutOp = (Id, Id, Vec<(Id, Scalar)>);

/// A POST handler.
pub type PostOp = Vec<(Id, Scalar)>;

/// A DELETE handler.
pub type DeleteOp = (Id, Vec<(Id, Scalar)>);

/// A user-defined operation.
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

impl OpDef {
    pub async fn call<I: IntoIterator<Item = (Id, State)>>(
        op_def: Vec<(Id, Scalar)>,
        txn: Txn,
        context: I,
    ) -> TCResult<State> {
        let capture = if let Some((id, _)) = op_def.last() {
            id.clone()
        } else {
            return Ok(State::default());
        };

        let context = context
            .into_iter()
            .chain(op_def.into_iter().map(|(id, s)| (id, State::Scalar(s))));

        Executor::new(txn, &State::default(), context)
            .capture(capture)
            .await
    }

    pub fn into_def(self) -> Vec<(Id, Scalar)> {
        match self {
            Self::Get((_, def)) => def,
            Self::Put((_, _, def)) => def,
            Self::Post(def) => def,
            Self::Delete((_, def)) => def,
        }
    }
}

pub struct OpDefVisitor;

impl OpDefVisitor {
    pub async fn visit_map_value<A: MapAccess>(
        class: OpDefType,
        map: &mut A,
    ) -> Result<OpDef, A::Error> {
        use OpDefType as ODT;

        match class {
            ODT::Get => {
                debug!("deserialize GET Op");

                let op = map.next_value(()).await?;
                Ok(OpDef::Get(op))
            }
            ODT::Put => {
                let op = map.next_value(()).await?;
                Ok(OpDef::Put(op))
            }
            ODT::Post => {
                let op = map.next_value(()).await?;
                Ok(OpDef::Post(op))
            }
            ODT::Delete => {
                let op = map.next_value(()).await?;
                Ok(OpDef::Delete(op))
            }
        }
    }
}

#[async_trait]
impl Visitor for OpDefVisitor {
    type Value = OpDef;

    fn expecting() -> &'static str {
        "an Op definition"
    }

    async fn visit_map<A: MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let err = || A::Error::custom("Expected an Op definition type, e.g. \"/state/op/get\"");

        let class = map.next_key::<String>(()).await?.ok_or_else(err)?;
        let class = TCPathBuf::from_str(&class).map_err(A::Error::custom)?;
        let class = OpDefType::from_path(&class).ok_or_else(err)?;

        Self::visit_map_value(class, &mut map).await
    }
}

#[async_trait]
impl FromStream for OpDef {
    type Context = ();

    async fn from_stream<D: Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
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

impl<'en> IntoStream<'en> for OpDef {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        let class = self.class().path().to_string();
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
