//! User-defined [`OpDef`]s

use std::fmt;
use std::iter;
use std::str::FromStr;

use async_hash::Hash;
use async_trait::async_trait;
use destream::de::{Decoder, Error, FromStream, MapAccess, Visitor};
use destream::en::{EncodeMap, Encoder, IntoStream, ToStream};
use log::debug;
use sha2::digest::{Digest, Output};

use tcgeneric::*;

use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler};
use crate::scalar::{Executor, Refer, Scalar};
use crate::state::State;

const PREFIX: PathLabel = path_label(&["state", "scalar", "op"]);

/// The [`Class`] of a user-defined [`OpDef`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum OpDefType {
    Get,
    Put,
    Post,
    Delete,
}

impl Class for OpDefType {}

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

impl OpDef {
    /// Replace references to `$self` with the given `path`.
    pub fn dereference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Get((key_name, form)) => Self::Get((key_name, dereference_self(form, path))),
            Self::Put((key_name, value_name, form)) => {
                Self::Put((key_name, value_name, dereference_self(form, path)))
            }
            Self::Post(form) => Self::Post(dereference_self(form, path)),
            Self::Delete((key_name, form)) => {
                Self::Delete((key_name, dereference_self(form, path)))
            }
        }
    }

    /// Iterate over the internal state assignments of this `OpDef`.
    pub fn form(&self) -> impl Iterator<Item = &(Id, Scalar)> {
        match self {
            Self::Get((_, form)) => form,
            Self::Put((_, _, form)) => form,
            Self::Post(form) => form,
            Self::Delete((_, form)) => form,
        }
        .iter()
    }

    /// Return the last assignment in this `OpDef`.
    pub fn last(&self) -> Option<&Id> {
        match self {
            Self::Get((_, form)) => form.last(),
            Self::Put((_, _, form)) => form.last(),
            Self::Post(form) => form.last(),
            Self::Delete((_, form)) => form.last(),
        }
        .map(|(id, _)| id)
    }

    /// Return `true` if this `OpDef` may execute a write operation to another service.
    pub fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        self.form()
            .map(|(_, provider)| provider)
            .any(|provider| provider.is_inter_service_write(cluster_path))
    }

    /// Consume this `OpDef` and return its internal state assignments.
    pub fn into_form(self) -> Vec<(Id, Scalar)> {
        match self {
            Self::Get((_, form)) => form,
            Self::Put((_, _, form)) => form,
            Self::Post(form) => form,
            Self::Delete((_, form)) => form,
        }
    }

    /// Return `true` if this is a write operation.
    pub fn is_write(&self) -> bool {
        match self {
            Self::Get(_) => false,
            Self::Put(_) => true,
            Self::Post(_) => false,
            Self::Delete(_) => true,
        }
    }

    /// Replace references to the given `path` with `$self`.
    pub fn reference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Get((key_name, form)) => Self::Get((key_name, reference_self(form, path))),
            Self::Put((key_name, value_name, form)) => {
                Self::Put((key_name, value_name, reference_self(form, path)))
            }
            Self::Post(form) => Self::Post(reference_self(form, path)),
            Self::Delete((key_name, form)) => Self::Delete((key_name, reference_self(form, path))),
        }
    }
}

impl<'a> Handler<'a> for OpDef {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if let OpDef::Get((key_name, op_def)) = *self {
            Some(Box::new(|txn, key| {
                Box::pin(async move {
                    let capture = if let Some((capture, _)) = op_def.last() {
                        capture.clone()
                    } else {
                        return Ok(State::default());
                    };

                    let key = State::from(key);
                    let data = iter::once((key_name, key)).chain(
                        op_def
                            .into_iter()
                            .map(|(id, provider)| (id, State::Scalar(provider))),
                    );

                    let executor: Executor<State> = Executor::new(txn, None, data);
                    executor.capture(capture).await
                })
            }))
        } else {
            None
        }
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if let OpDef::Put((key_name, value_name, op_def)) = *self {
            Some(Box::new(|txn, key, value| {
                Box::pin(async move {
                    let capture = if let Some((capture, _)) = op_def.last() {
                        capture.clone()
                    } else {
                        return Ok(());
                    };

                    let key = State::from(key);
                    let args = std::array::IntoIter::new([(key_name, key), (value_name, value)]);
                    let op_def = op_def
                        .into_iter()
                        .map(|(id, provider)| (id, State::Scalar(provider)));

                    let data = args.chain(op_def);
                    let executor: Executor<State> = Executor::new(txn, None, data);
                    executor.capture(capture).await?;
                    Ok(())
                })
            }))
        } else {
            None
        }
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if let OpDef::Post(op_def) = *self {
            Some(Box::new(|txn, params| {
                Box::pin(async move {
                    let capture = if let Some((capture, _)) = op_def.last() {
                        capture.clone()
                    } else {
                        return Ok(State::default());
                    };

                    let op_def = op_def
                        .into_iter()
                        .map(|(id, provider)| (id, State::Scalar(provider)));

                    let data = params.into_iter().chain(op_def);
                    let executor: Executor<State> = Executor::new(txn, None, data);
                    executor.capture(capture).await
                })
            }))
        } else {
            None
        }
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if let OpDef::Get((key_name, op_def)) = *self {
            Some(Box::new(|txn, key| {
                Box::pin(async move {
                    let capture = if let Some((capture, _)) = op_def.last() {
                        capture.clone()
                    } else {
                        return Ok(());
                    };

                    let key = State::from(key);
                    let data = iter::once((key_name, key)).chain(
                        op_def
                            .into_iter()
                            .map(|(id, provider)| (id, State::Scalar(provider))),
                    );

                    let executor: Executor<State> = Executor::new(txn, None, data);
                    executor.capture(capture).await?;
                    Ok(())
                })
            }))
        } else {
            None
        }
    }
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

impl<D: Digest> Hash<D> for OpDef {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a OpDef {
    fn hash(self) -> Output<D> {
        match self {
            OpDef::Get(get) => Hash::<D>::hash(get),
            OpDef::Put(put) => Hash::<D>::hash(put),
            OpDef::Post(post) => Hash::<D>::hash(post),
            OpDef::Delete(delete) => Hash::<D>::hash(delete),
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

impl fmt::Debug for OpDef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
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

fn dereference_self(form: Vec<(Id, Scalar)>, path: &TCPathBuf) -> Vec<(Id, Scalar)> {
    form.into_iter()
        .map(|(id, scalar)| (id, scalar.dereference_self(path)))
        .collect()
}

fn reference_self(form: Vec<(Id, Scalar)>, path: &TCPathBuf) -> Vec<(Id, Scalar)> {
    form.into_iter()
        .map(|(id, scalar)| (id, scalar.reference_self(path)))
        .collect()
}
