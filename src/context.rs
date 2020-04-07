use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::de;
use serde::ser;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::state::TCState;
use crate::transaction::{Request, Transaction};

const LINK_BLACKLIST: [&str; 11] = ["..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "="];

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Clone, Hash)]
pub enum TCResponse {
    Exe(Request),
    State(Arc<TCState>),
    Value(TCValue),
}

impl TCResponse {
    pub fn to_state(&self) -> TCResult<Arc<TCState>> {
        match self {
            TCResponse::State(state) => Ok(state.clone()),
            other => Err(error::bad_request("Expected state but found", other)),
        }
    }
}

impl fmt::Display for TCResponse {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCResponse::Exe(_exe) => write!(f, "(executable)"),
            TCResponse::State(state) => write!(f, "{}", state),
            TCResponse::Value(value) => write!(f, "{}", value),
        }
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize, Hash, Eq, PartialEq)]
pub struct Link {
    to: String,
    segments: Vec<String>,
}

impl Link {
    fn _validate(to: &str) -> TCResult<()> {
        for pattern in LINK_BLACKLIST.iter() {
            if to.contains(pattern) {
                return Err(error::bad_request(
                    "Tinychain links do not allow this pattern",
                    pattern,
                ));
            }
        }

        if !to.starts_with('/') {
            Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                to,
            ))
        } else if to != "/" && to.ends_with('/') {
            Err(error::bad_request("Trailing slash is not allowed", to))
        } else if Regex::new(r"\s").unwrap().find(to).is_some() {
            Err(error::bad_request(
                "Tinychain links do not allow whitespace",
                to,
            ))
        } else {
            Ok(())
        }
    }

    pub fn to(destination: &str) -> TCResult<Link> {
        Link::_validate(destination)?;

        Ok(Link {
            to: destination.to_string(),
            segments: destination[1..].split('/').map(|s| s.to_string()).collect(),
        })
    }

    pub fn as_str(&self) -> &str {
        &self.to
    }

    pub fn append(&self, suffix: Link) -> Link {
        Link::to(&format!("{}{}", self.to, suffix.to)).unwrap()
    }

    pub fn from(&self, prefix: &str) -> TCResult<Link> {
        if prefix.ends_with('/') {
            return Err(error::bad_request("Link prefix cannot end in a /", prefix));
        }
        if !self.to.starts_with(prefix) {
            return Err(error::bad_request(
                &format!("Cannot link {} from", self),
                prefix,
            ));
        }

        Link::to(&self.to[prefix.len()..])
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn segment(&self, i: usize) -> Link {
        let to = format!("/{}", self.segments[i]);
        Link {
            to,
            segments: vec![self.segments[i].clone()],
        }
    }

    #[allow(dead_code)]
    fn deserialize<'de, D>(deserializer: D) -> Result<Link, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        Link::to(s).map_err(de::Error::custom)
    }

    #[allow(dead_code)]
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        s.serialize_str(self.as_str())
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.to)
    }
}

impl IntoIterator for Link {
    type Item = Link;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut segments = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            segments.push(self.segment(i));
        }
        segments.into_iter()
    }
}

impl<Idx> std::ops::Index<Idx> for Link
where
    Idx: std::slice::SliceIndex<[String]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.segments[index]
    }
}

#[derive(Clone, Deserialize, Hash, Eq, PartialEq)]
pub enum TCValue {
    Bytes(Vec<u8>),
    Int32(i32),
    Link(Link),
    r#String(String),
}

impl TCValue {
    pub fn to_bytes(&self) -> TCResult<Vec<u8>> {
        match self {
            TCValue::Bytes(b) => Ok(b.clone()),
            other => Err(error::bad_request("Expected bytes but found", other)),
        }
    }

    pub fn to_link(&self) -> TCResult<Link> {
        match self {
            TCValue::Link(l) => Ok(l.clone()),
            other => Err(error::bad_request("Expected a Link but found", other)),
        }
    }
}

impl Serialize for TCValue {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: ser::Serializer,
    {
        match self {
            TCValue::Bytes(b) => s.serialize_bytes(b),
            TCValue::Int32(i) => s.serialize_i32(*i),
            TCValue::Link(l) => l.serialize(s),
            TCValue::r#String(v) => s.serialize_str(v),
        }
    }
}

impl fmt::Debug for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for TCValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCValue::Bytes(b) => write!(f, "binary of length {}", b.len()),
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
            TCValue::r#String(s) => write!(f, "string: {}", s),
        }
    }
}

#[async_trait]
pub trait TCContext: Send + Sync {
    async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<TCResponse>;

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, value: TCValue) -> TCResult<()>;
}

#[async_trait]
pub trait TCExecutable: Send + Sync {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<TCResponse>;
}
