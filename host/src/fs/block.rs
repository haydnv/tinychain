use std::fs::Metadata;
use std::io;
use std::path::Path;

use async_trait::async_trait;
use safecast::AsType;
use tokio::fs;

#[cfg(feature = "tensor")]
use afarray::Array;
use tc_btree::Node;
use tc_value::Value;

use crate::chain::ChainBlock;

use super::file_ext;

pub enum CacheBlock {
    BTree(Node),
    Chain(ChainBlock),
    #[cfg(feature = "tensor")]
    Tensor(Array),
    Value(Value),
}

#[async_trait]
impl freqfs::FileLoad for CacheBlock {
    async fn load(path: &Path, file: fs::File, metadata: Metadata) -> Result<Self, io::Error> {
        todo!()
    }

    async fn save(&self, file: &mut fs::File) -> Result<u64, io::Error> {
        todo!()
    }
}

impl AsType<Node> for CacheBlock {
    fn as_type(&self) -> Option<&Node> {
        if let Self::BTree(node) = self {
            Some(node)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut Node> {
        if let Self::BTree(node) = self {
            Some(node)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<Node> {
        if let Self::BTree(node) = self {
            Some(node)
        } else {
            None
        }
    }
}

impl AsType<ChainBlock> for CacheBlock {
    fn as_type(&self) -> Option<&ChainBlock> {
        if let Self::Chain(block) = self {
            Some(block)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut ChainBlock> {
        if let Self::Chain(block) = self {
            Some(block)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<ChainBlock> {
        if let Self::Chain(block) = self {
            Some(block)
        } else {
            None
        }
    }
}

#[cfg(feature = "tensor")]
impl AsType<Array> for CacheBlock {
    fn as_type(&self) -> Option<&Array> {
        if let Self::Tensor(array) = self {
            Some(array)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut Array> {
        if let Self::Tensor(array) = self {
            Some(array)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<Array> {
        if let Self::Tensor(array) = self {
            Some(array)
        } else {
            None
        }
    }
}

impl AsType<Value> for CacheBlock {
    fn as_type(&self) -> Option<&Value> {
        if let Self::Value(value) = self {
            Some(value)
        } else {
            None
        }
    }

    fn as_type_mut(&mut self) -> Option<&mut Value> {
        if let Self::Value(value) = self {
            Some(value)
        } else {
            None
        }
    }

    fn into_type(self) -> Option<Value> {
        if let Self::Value(value) = self {
            Some(value)
        } else {
            None
        }
    }
}

impl From<Node> for CacheBlock {
    fn from(node: Node) -> Self {
        Self::BTree(node)
    }
}

impl From<ChainBlock> for CacheBlock {
    fn from(block: ChainBlock) -> Self {
        Self::Chain(block)
    }
}

#[cfg(feature = "tensor")]
impl From<Array> for CacheBlock {
    fn from(array: Array) -> Self {
        Self::Tensor(array)
    }
}

impl From<Value> for CacheBlock {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}
