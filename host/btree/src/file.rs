use std::fmt;
use std::iter::FromIterator;
use std::ops::Deref;

use async_trait::async_trait;
use destream::{de, en};
use futures::TryFutureExt;

use tc_transact::fs::{BlockData, BlockId};
use tc_value::Value;
use tcgeneric::Tuple;

type NodeId = BlockId;

#[derive(Clone)]
struct NodeKey {
    deleted: bool,
    value: Vec<Value>,
}

impl NodeKey {
    fn new(value: Vec<Value>) -> Self {
        Self {
            deleted: false,
            value,
        }
    }
}

impl Deref for NodeKey {
    type Target = [Value];

    fn deref(&self) -> &[Value] {
        &self.value
    }
}

#[async_trait]
impl de::FromStream for NodeKey {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|(deleted, value)| Self { deleted, value })
            .await
    }
}

impl<'en> en::ToStream<'en> for NodeKey {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream((&self.deleted, &self.value), encoder)
    }
}

#[cfg(debug_assertions)]
impl fmt::Display for NodeKey {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BTree node key: {}{}",
            Value::from_iter(self.value.to_vec()),
            if self.deleted { " (DELETED)" } else { "" }
        )
    }
}

#[derive(Clone)]
pub struct Node {
    leaf: bool,
    keys: Vec<NodeKey>,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    rebalance: bool, // TODO: implement rebalancing to clear deleted values
}

impl Node {
    fn new(leaf: bool, parent: Option<NodeId>) -> Node {
        Node {
            leaf,
            keys: vec![],
            parent,
            children: vec![],
            rebalance: false,
        }
    }
}

impl<'en> BlockData<'en> for Node {
    fn ext() -> &'static str {
        "node"
    }
}

#[async_trait]
impl de::FromStream for Node {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|(leaf, keys, parent, children, rebalance)| Self {
                leaf,
                keys,
                parent,
                children,
                rebalance,
            })
            .await
    }
}

impl<'en> en::ToStream<'en> for Node {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(
            (
                &self.leaf,
                &self.keys,
                &self.parent,
                &self.children,
                &self.rebalance,
            ),
            encoder,
        )
    }
}

#[cfg(debug_assertions)]
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.leaf {
            writeln!(f, "leaf node:")?;
        } else {
            writeln!(f, "non-leaf node:")?;
        }

        write!(
            f,
            "\tkeys: {}",
            Tuple::<NodeKey>::from_iter(self.keys.iter().cloned())
        )?;
        write!(f, "\t {} children", self.children.len())
    }
}

#[derive(Clone)]
pub struct BTreeFile {}
