use std::io;

use async_trait::async_trait;
use bytes::Bytes;
use destream::en;
use futures::TryStreamExt;
use get_size::GetSize;
use get_size_derive::*;
use safecast::{as_type, AsType};
use tokio::fs;
use tokio_util::io::StreamReader;

use tc_chain::ChainBlock;
use tc_collection::{btree, tensor};
use tc_scalar::Scalar;
use tc_value::Link;
use tcgeneric::Map;

/// A block of a [`tensor::Dense`] tensor
#[derive(Clone, GetSize)]
pub enum DenseBuffer {
    F32(tensor::Buffer<f32>),
    F64(tensor::Buffer<f64>),
    I16(tensor::Buffer<i16>),
    I32(tensor::Buffer<i32>),
    I64(tensor::Buffer<i64>),
    U8(tensor::Buffer<u8>),
    U16(tensor::Buffer<u16>),
    U32(tensor::Buffer<u32>),
    U64(tensor::Buffer<u64>),
}

as_type!(DenseBuffer, F32, tensor::Buffer<f32>);
as_type!(DenseBuffer, F64, tensor::Buffer<f64>);
as_type!(DenseBuffer, I16, tensor::Buffer<i16>);
as_type!(DenseBuffer, I32, tensor::Buffer<i32>);
as_type!(DenseBuffer, I64, tensor::Buffer<i64>);
as_type!(DenseBuffer, U8, tensor::Buffer<u8>);
as_type!(DenseBuffer, U16, tensor::Buffer<u16>);
as_type!(DenseBuffer, U32, tensor::Buffer<u32>);
as_type!(DenseBuffer, U64, tensor::Buffer<u64>);

impl<'en> en::ToStream<'en> for DenseBuffer {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::F32(this) => this.to_stream(encoder),
            Self::F64(this) => this.to_stream(encoder),
            Self::I16(this) => this.to_stream(encoder),
            Self::I32(this) => this.to_stream(encoder),
            Self::I64(this) => this.to_stream(encoder),
            Self::U8(this) => this.to_stream(encoder),
            Self::U16(this) => this.to_stream(encoder),
            Self::U32(this) => this.to_stream(encoder),
            Self::U64(this) => this.to_stream(encoder),
        }
    }
}

/// A cached filesystem block.
#[derive(Clone, GetSize)]
pub enum CacheBlock {
    BTree(btree::Node),
    Chain(ChainBlock),
    Class((Link, Map<Scalar>)),
    Library(Map<Scalar>),
    Sparse(tensor::Node),
    Dense(DenseBuffer),
}

#[async_trait]
impl<'en> tc_transact::fs::FileSave<'en> for CacheBlock {
    async fn save(&'en self, file: &mut fs::File) -> Result<u64, io::Error> {
        match self {
            Self::BTree(node) => persist(node, file).await,
            Self::Chain(block) => persist(block, file).await,
            Self::Class(class) => persist(class, file).await,
            Self::Library(library) => persist(library, file).await,
            Self::Dense(dense) => persist(dense, file).await,
            Self::Sparse(sparse) => persist(sparse, file).await,
        }
    }
}

as_type!(CacheBlock, BTree, btree::Node);
as_type!(CacheBlock, Chain, ChainBlock);
as_type!(CacheBlock, Class, (Link, Map<Scalar>));
as_type!(CacheBlock, Library, Map<Scalar>);
as_type!(CacheBlock, Sparse, tensor::Node);

macro_rules! as_dense_type {
    ($t:ty) => {
        impl AsType<tensor::Buffer<$t>> for CacheBlock {
            fn as_type(&self) -> Option<&tensor::Buffer<$t>> {
                if let Self::Dense(block) = self {
                    block.as_type()
                } else {
                    None
                }
            }

            fn as_type_mut(&mut self) -> Option<&mut tensor::Buffer<$t>> {
                if let Self::Dense(block) = self {
                    block.as_type_mut()
                } else {
                    None
                }
            }

            fn into_type(self) -> Option<tensor::Buffer<$t>> {
                if let Self::Dense(block) = self {
                    block.into_type()
                } else {
                    None
                }
            }
        }

        impl From<tensor::Buffer<$t>> for CacheBlock {
            fn from(buffer: tensor::Buffer<$t>) -> Self {
                Self::Dense(buffer.into())
            }
        }
    };
}

as_dense_type!(f32);
as_dense_type!(f64);
as_dense_type!(i16);
as_dense_type!(i32);
as_dense_type!(i64);
as_dense_type!(u8);
as_dense_type!(u16);
as_dense_type!(u32);
as_dense_type!(u64);

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
