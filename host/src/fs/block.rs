use std::fs::Metadata;
use std::io;
use std::path::Path;

use async_trait::async_trait;
use bytes::Bytes;
use destream::en;
use futures::{TryFutureExt, TryStreamExt};
use get_size::GetSize;
use get_size_derive::*;
use safecast::as_type;
use tokio::fs;
use tokio_util::io::StreamReader;

#[cfg(any(feature = "btree", feature = "table"))]
use tc_btree::Node;
#[cfg(feature = "tensor")]
use tc_tensor::Array;

use crate::chain::ChainBlock;
use crate::cluster::library;
use crate::object::InstanceClass;

use super::file_ext;

/// A cached filesystem block.
#[derive(Clone, GetSize)]
pub enum CacheBlock {
    #[cfg(any(feature = "btree", feature = "table"))]
    BTree(Node),
    Chain(ChainBlock),
    Class(InstanceClass),
    Library(library::Version),
    #[cfg(feature = "tensor")]
    Tensor(Array),
}

#[async_trait]
impl freqfs::FileLoad for CacheBlock {
    async fn load(path: &Path, file: fs::File, _metadata: Metadata) -> Result<Self, io::Error> {
        match file_ext(path) {
            #[cfg(feature = "tensor")]
            Some("array") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::Tensor)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            Some("chain_block") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::Chain)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            Some("class") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::Class)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            Some("lib") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::Library)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            #[cfg(any(feature = "btree", feature = "table"))]
            Some("node") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::BTree)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            Some(other) => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unrecognized block extension: {}", other),
            )),

            None => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("block name is missing an extension: {:?}", path.file_name()),
            )),
        }
    }

    async fn save(&self, file: &mut fs::File) -> Result<u64, io::Error> {
        match self {
            #[cfg(any(feature = "btree", feature = "table"))]
            Self::BTree(node) => persist(node, file).await,
            Self::Chain(block) => persist(block, file).await,
            Self::Class(class) => persist(class, file).await,
            Self::Library(library) => persist(library, file).await,
            #[cfg(feature = "tensor")]
            Self::Tensor(array) => persist(array, file).await,
        }
    }
}

#[cfg(any(feature = "btree", feature = "table"))]
as_type!(CacheBlock, BTree, Node);
as_type!(CacheBlock, Chain, ChainBlock);
as_type!(CacheBlock, Class, InstanceClass);
as_type!(CacheBlock, Library, library::Version);
#[cfg(feature = "tensor")]
as_type!(CacheBlock, Tensor, Array);

async fn persist<'en, T: en::ToStream<'en>>(
    data: &'en T,
    file: &mut fs::File,
) -> Result<u64, io::Error> {
    let encoded = tbon::en::encode(data)
        .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))?;

    let mut reader = StreamReader::new(
        encoded
            .map_ok(Bytes::from)
            .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause)),
    );

    tokio::io::copy(&mut reader, file).await
}
