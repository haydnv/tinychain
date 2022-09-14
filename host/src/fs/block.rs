use std::fs::Metadata;
use std::io;
use std::path::Path;

use async_trait::async_trait;
use bytes::Bytes;
use destream::en;
use futures::{TryFutureExt, TryStreamExt};
use safecast::{as_type, AsType};
use tokio::fs;
use tokio_util::io::StreamReader;

use tc_btree::Node;
#[cfg(feature = "tensor")]
use tc_tensor::Array;

use crate::chain::ChainBlock;
use crate::scalar::Scalar;

use super::file_ext;

/// A cached filesystem block.
#[derive(Clone)]
pub enum CacheBlock {
    BTree(Node),
    Chain(ChainBlock),
    #[cfg(feature = "tensor")]
    Tensor(Array),
    Scalar(Scalar),
}

#[async_trait]
impl freqfs::FileLoad for CacheBlock {
    async fn load(path: &Path, file: fs::File, _metadata: Metadata) -> Result<Self, io::Error> {
        match file_ext(path) {
            Some("node") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::BTree)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            Some("chain_block") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::Chain)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            #[cfg(feature = "tensor")]
            Some("array") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::Tensor)
                    .map_err(|cause| io::Error::new(io::ErrorKind::InvalidData, cause))
                    .await
            }

            Some("scalar") => {
                tbon::de::read_from((), file)
                    .map_ok(Self::Scalar)
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
            Self::BTree(node) => persist(node, file).await,
            Self::Chain(block) => persist(block, file).await,
            #[cfg(feature = "tensor")]
            Self::Tensor(array) => persist(array, file).await,
            Self::Scalar(scalar) => persist(scalar, file).await,
        }
    }
}

as_type!(CacheBlock, BTree, Node);
as_type!(CacheBlock, Chain, ChainBlock);
as_type!(CacheBlock, Scalar, Scalar);
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
