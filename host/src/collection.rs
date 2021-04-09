use std::fmt;

use tcgeneric::{path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf};

use crate::fs;
use crate::txn::Txn;

const PREFIX: PathLabel = path_label(&["state", "collection"]);

pub use tc_btree::BTreeType;

pub type BTree = tc_btree::BTree<fs::File<tc_btree::Node>, fs::Dir, Txn>;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum CollectionType {
    BTree(BTreeType),
}

impl Class for CollectionType {}

impl NativeClass for CollectionType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() > 2 && &path[0..2] == &PREFIX[..] {
            match path[2].as_str() {
                "btree" => BTreeType::from_path(path).map(Self::BTree),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::BTree(btree) => btree.path(),
        }
    }
}

impl fmt::Display for CollectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btt) => fmt::Display::fmt(btt, f),
        }
    }
}

#[derive(Clone)]
pub enum Collection {
    BTree(BTree),
}

impl Instance for Collection {
    type Class = CollectionType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => CollectionType::BTree(btree.class()),
        }
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(btree) => fmt::Display::fmt(btree, f),
        }
    }
}
