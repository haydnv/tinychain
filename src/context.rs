use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde;
use serde::de;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::state::block::Block;
use crate::state::chain::Chain;
use crate::state::table::Table;
use crate::transaction::Transaction;

const LINK_BLACKLIST: [&str; 11] = ["..", "~", "$", "&", "?", "|", "{", "}", "//", ":", "="];

pub type TCResult<T> = Result<T, error::TCError>;

#[derive(Clone, Deserialize, Serialize, Hash, Eq, PartialEq)]
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

    pub fn new() -> Link {
        Link {
            to: "/".to_string(),
            segments: vec![],
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

    pub fn split(&self, idx: usize) -> TCResult<(Link, Link)> {
        if idx > self.segments.len() {
            return Err(error::bad_request(
                &format!(
                    "Tried to read segment {} of a link with {} segments",
                    idx,
                    self.segments.len()
                ),
                self,
            ));
        }

        let left = Link::to(&format!("/{}", &self.segments[..idx].join("/")))?;
        let right = Link::to(&format!("/{}", &self.segments[(idx + 1)..].join("/")))?;

        Ok((left, right))
    }
}

fn deserialize_link<'de, D>(deserializer: D) -> Result<Link, D::Error>
where
    D: de::Deserializer<'de>,
{
    let s: &str = de::Deserialize::deserialize(deserializer)?;
    Link::to(s).map_err(de::Error::custom)
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

#[derive(Clone, Deserialize, Serialize, Hash)]
pub enum TCValue {
    None,
    Bytes(Vec<u8>),
    Int32(i32),

    #[serde(deserialize_with = "deserialize_link")]
    Link(Link),
    r#String(String),
    Vector(Vec<TCValue>),
}

impl TCValue {
    pub fn from_bytes(b: Vec<u8>) -> TCValue {
        TCValue::Bytes(b)
    }

    pub fn from_string(s: &str) -> TCValue {
        TCValue::r#String(s.to_string())
    }

    pub fn to_bytes(&self) -> TCResult<Vec<u8>> {
        match self {
            TCValue::Bytes(b) => Ok(b.clone()),
            other => Err(error::bad_request("Expected bytes but found", other)),
        }
    }

    pub fn to_link(&self) -> TCResult<Link> {
        match self {
            TCValue::Link(l) => Ok(l.clone()),
            other => Err(error::bad_request("Expected link but found", other)),
        }
    }

    pub fn to_string(&self) -> TCResult<String> {
        match self {
            TCValue::r#String(s) => Ok(s.clone()),
            other => Err(error::bad_request("Expected string but found", other)),
        }
    }

    pub fn to_vec(&self) -> TCResult<Vec<TCValue>> {
        match self {
            TCValue::Vector(vec) => Ok(vec.clone()),
            other => Err(error::bad_request("Expected vector but found", other)),
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
            TCValue::None => write!(f, "None"),
            TCValue::Bytes(b) => write!(f, "binary of length {}", b.len()),
            TCValue::Int32(i) => write!(f, "Int32: {}", i),
            TCValue::Link(l) => write!(f, "Link: {}", l),
            TCValue::r#String(s) => write!(f, "string: {}", s),
            TCValue::Vector(v) => write!(f, "vector of length {}", v.len()),
        }
    }
}

#[derive(Hash)]
pub enum TCState {
    Block(Arc<Block>),
    Chain(Arc<Chain>),
    Table(Arc<Table>),
    Value(TCValue),
}

impl TCState {
    pub fn from_block(block: Arc<Block>) -> Arc<TCState> {
        Arc::new(TCState::Block(block))
    }

    pub fn from_chain(chain: Arc<Chain>) -> Arc<TCState> {
        Arc::new(TCState::Chain(chain))
    }

    pub fn from_value(value: TCValue) -> Arc<TCState> {
        Arc::new(TCState::Value(value))
    }

    pub fn to_block(self: Arc<Self>) -> TCResult<Arc<Block>> {
        match &*self {
            TCState::Block(block) => Ok(block.clone()),
            other => Err(error::bad_request("Expected block but found", other)),
        }
    }

    pub fn to_chain(self: Arc<Self>) -> TCResult<Arc<Chain>> {
        match &*self {
            TCState::Chain(chain) => Ok(chain.clone()),
            other => Err(error::bad_request("Expected chain but found", other)),
        }
    }

    pub fn to_value(self: Arc<Self>) -> TCResult<TCValue> {
        match &*self {
            TCState::Value(val) => Ok(val.clone()),
            other => Err(error::bad_request("Expected value but found", other)),
        }
    }
}

impl fmt::Display for TCState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TCState::Block(_) => write!(f, "(block)"),
            TCState::Chain(_) => write!(f, "(chain)"),
            TCState::Table(_) => write!(f, "(table)"),
            TCState::Value(v) => write!(f, "value: {}", v),
        }
    }
}

#[async_trait]
impl TCContext for TCState {
    async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<Arc<TCState>> {
        match &*self {
            TCState::Block(b) => b.clone().get(txn, path).await,
            TCState::Chain(c) => c.clone().get(txn, path).await,
            TCState::Table(t) => t.clone().get(txn, path).await,
            TCState::Value(_) => Err(error::method_not_allowed(path)),
        }
    }

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, value: TCValue) -> TCResult<()> {
        match &*self {
            TCState::Block(b) => b.clone().put(txn, value).await,
            TCState::Chain(c) => c.clone().put(txn, value).await,
            TCState::Table(t) => t.clone().put(txn, value).await,
            TCState::Value(_) => Err(error::method_not_allowed("TCValue")),
        }
    }
}

#[async_trait]
pub trait TCContext: Send + Sync {
    async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<Arc<TCState>>;

    async fn put(self: Arc<Self>, txn: Arc<Transaction>, value: TCValue) -> TCResult<()>;
}

#[async_trait]
pub trait TCExecutable: Send + Sync {
    async fn post(self: Arc<Self>, txn: Arc<Transaction>, method: Link) -> TCResult<Arc<TCState>>;
}

#[async_trait]
pub trait TCObject: TCContext + TCExecutable {}
