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

use tc_collection::Node;
#[cfg(feature = "collection")]
use tc_tensor::Array;

use crate::chain::ChainBlock;
use crate::cluster::library;
use crate::object::InstanceClass;

/// A transactional directory
pub type Dir = tc_transact::fs::Dir<CacheBlock>;

/// An entry in a transactional directory
pub type DirEntry = tc_transact::fs::DirEntry<CacheBlock>;

/// A transactional file
pub type File<N, B> = tc_transact::fs::File<CacheBlock, N, B>;

/// A cached filesystem block.
#[derive(Clone, GetSize)]
pub enum CacheBlock {
    BTree(Node),
    Chain(ChainBlock),
    Class(InstanceClass),
    Library(library::Version),
    #[cfg(feature = "collection")]
    Tensor(Array),
}

#[async_trait]
impl<'en> freqfs::FileSave<'en> for CacheBlock {
    async fn save(&'en self, file: &mut fs::File) -> Result<u64, io::Error> {
        match self {
            Self::BTree(node) => persist(node, file).await,
            Self::Chain(block) => persist(block, file).await,
            Self::Class(class) => persist(class, file).await,
            Self::Library(library) => persist(library, file).await,
            #[cfg(feature = "collection")]
            Self::Tensor(array) => persist(array, file).await,
        }
    }
}

as_type!(CacheBlock, BTree, Node);
as_type!(CacheBlock, Chain, ChainBlock);
as_type!(CacheBlock, Class, InstanceClass);
as_type!(CacheBlock, Library, library::Version);
#[cfg(feature = "collection")]
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
