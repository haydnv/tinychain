use std::fmt;
use std::pin::Pin;

use destream::{de, en};
use futures::future::{self, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use futures::try_join;
use ha_ndarray::{Buffer, CDatatype, NDArray, NDArrayRead, Queue};
use rayon::prelude::*;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::lock::PermitRead;
use tc_transact::{Transaction, TxnId};
use tc_value::{DType, Number, ValueType};
use tcgeneric::{NativeClass, TCPathBuf};

use super::dense::{DenseAccess, DenseCacheFile, DenseInstance, DenseTensor};
use super::sparse::Node;
use super::{Coord, Dense, Range, Sparse, Tensor, TensorInstance, TensorPermitRead};

type Blocks<T> = Pin<Box<dyn Stream<Item = Vec<T>> + Send>>;

enum DenseViewBlocks {
    Bool(Blocks<bool>),
    F32(Blocks<f32>),
    F64(Blocks<f64>),
    I16(Blocks<i16>),
    I32(Blocks<i32>),
    I64(Blocks<i64>),
    U8(Blocks<u8>),
    U16(Blocks<u16>),
    U32(Blocks<u32>),
    U64(Blocks<u64>),
}

impl DenseViewBlocks {
    async fn read_from<Txn, FE>(tensor: super::DenseView<Txn, FE>, txn_id: TxnId) -> TCResult<Self>
    where
        Txn: Transaction<FE>,
        FE: DenseCacheFile + AsType<Node> + Clone,
    {
        let _permit = tensor.read_permit(txn_id, Range::default()).await?;

        match tensor {
            super::DenseView::Bool(tensor) => {
                let access = tensor.into_inner();
                let blocks = access.read_blocks(txn_id).await?;
                let blocks = blocks
                    .map(move |result| {
                        let block = result?;
                        let queue = Queue::new(block.context().clone(), block.size())?;
                        let buffer = block.read(&queue)?.to_slice()?.into_vec();
                        TCResult::Ok(buffer.into_iter().map(|i| i != 0).collect::<Vec<bool>>())
                    })
                    .take_while(|result| future::ready(result.is_ok()))
                    .map(|result| result.expect("buffer"));

                Ok(Self::Bool(Box::pin(blocks)))
            }
            super::DenseView::C32((re, im)) => {
                read_from_complex(txn_id, re, im).map_ok(Self::F32).await
            }
            super::DenseView::C64((re, im)) => {
                read_from_complex(txn_id, re, im).map_ok(Self::F64).await
            }
            super::DenseView::F32(tensor) => read_from_real(txn_id, tensor).map_ok(Self::F32).await,
            super::DenseView::F64(tensor) => read_from_real(txn_id, tensor).map_ok(Self::F64).await,
            super::DenseView::I16(tensor) => read_from_real(txn_id, tensor).map_ok(Self::I16).await,
            super::DenseView::I32(tensor) => read_from_real(txn_id, tensor).map_ok(Self::I32).await,
            super::DenseView::I64(tensor) => read_from_real(txn_id, tensor).map_ok(Self::I64).await,
            super::DenseView::U8(tensor) => read_from_real(txn_id, tensor).map_ok(Self::U8).await,
            super::DenseView::U16(tensor) => read_from_real(txn_id, tensor).map_ok(Self::U16).await,
            super::DenseView::U32(tensor) => read_from_real(txn_id, tensor).map_ok(Self::U32).await,
            super::DenseView::U64(tensor) => read_from_real(txn_id, tensor).map_ok(Self::U64).await,
        }
    }
}

#[inline]
async fn read_from_complex<Txn, FE, T>(
    txn_id: TxnId,
    re: DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>,
    im: DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>,
) -> TCResult<Blocks<T>>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node> + Clone,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    let re = re.into_inner();
    let im = im.into_inner();
    let (re, im) = try_join!(re.read_blocks(txn_id), im.read_blocks(txn_id))?;

    let re = re.map(move |result| {
        let block = result?;
        let queue = Queue::new(block.context().clone(), block.size())?;
        block
            .read(&queue)
            .and_then(|buffer| buffer.to_slice())
            .map(|slice| slice.into_vec())
            .map_err(TCError::from)
    });

    let im = im.map(move |result| {
        let block = result?;
        let queue = Queue::new(block.context().clone(), block.size())?;
        block
            .read(&queue)
            .and_then(|buffer| buffer.to_slice())
            .map(|slice| slice.into_vec())
            .map_err(TCError::from)
    });

    let blocks = re
        .zip(im)
        .map(|(re, im)| TCResult::Ok((re?, im?)))
        .map_ok(|(re, im)| {
            re.into_par_iter()
                .zip(im)
                .map(|(r, i)| [r, i].into_par_iter())
                .flatten()
                .collect::<Vec<T>>()
        })
        .take_while(|result| future::ready(result.is_ok()))
        .map(|result| result.expect("buffer"));

    Ok(Box::pin(blocks))
}

#[inline]
async fn read_from_real<Txn, FE, T>(
    txn_id: TxnId,
    tensor: DenseTensor<Txn, FE, DenseAccess<Txn, FE, T>>,
) -> TCResult<Blocks<T>>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node> + Clone,
    T: CDatatype + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    let access = tensor.into_inner();
    let blocks = access.read_blocks(txn_id).await?;
    let blocks = blocks
        .map(move |result| {
            let block = result?;
            let queue = Queue::new(block.context().clone(), block.size())?;
            block
                .read(&queue)
                .and_then(|buffer| buffer.to_slice())
                .map(|slice| slice.into_vec())
                .map_err(TCError::from)
        })
        .take_while(|result| {
            future::ready({
                match result {
                    Ok(_) => true,
                    Err(cause) => {
                        #[cfg(debug_assertions)]
                        panic!("failed to read dense block! {cause}");

                        #[cfg(not(debug_assertions))]
                        {
                            log::error!("failed to read dense block! {cause}");
                            false
                        }
                    }
                }
            })
        })
        .map(|result| result.expect("buffer"));

    Ok(Box::pin(blocks))
}

impl<'en> en::IntoStream<'en> for DenseViewBlocks {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Bool(blocks) => encoder.encode_array_bool(blocks),
            Self::F32(blocks) => encoder.encode_array_f32(blocks),
            Self::F64(blocks) => encoder.encode_array_f64(blocks),
            Self::I16(blocks) => encoder.encode_array_i16(blocks),
            Self::I32(blocks) => encoder.encode_array_i32(blocks),
            Self::I64(blocks) => encoder.encode_array_i64(blocks),
            Self::U8(blocks) => encoder.encode_array_u8(blocks),
            Self::U16(blocks) => encoder.encode_array_u16(blocks),
            Self::U32(blocks) => encoder.encode_array_u32(blocks),
            Self::U64(blocks) => encoder.encode_array_u64(blocks),
        }
    }
}

pub struct DenseView {
    _permit: Vec<PermitRead<Range>>,
    schema: (TCPathBuf, Vec<u64>),
    elements: DenseViewBlocks,
}

impl DenseView {
    pub async fn read_from<Txn, FE>(tensor: Dense<Txn, FE>, txn_id: TxnId) -> TCResult<Self>
    where
        Txn: Transaction<FE>,
        FE: DenseCacheFile + AsType<Node> + Clone,
    {
        let tensor = tensor.into_view();
        let permit = tensor.read_permit(txn_id, Range::default()).await?;

        let schema = (
            ValueType::Number(tensor.dtype()).path(),
            tensor.shape().to_vec(),
        );

        let elements = DenseViewBlocks::read_from(tensor, txn_id).await?;

        Ok(Self {
            schema,
            elements,
            _permit: permit,
        })
    }
}

impl<'en> en::IntoStream<'en> for DenseView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeSeq;

        let mut seq = encoder.encode_seq(Some(2))?;
        seq.encode_element(self.schema)?;
        seq.encode_element(self.elements)?;
        seq.end()
    }
}

pub struct SparseView {
    _permit: Vec<PermitRead<Range>>,
    schema: (TCPathBuf, Vec<u64>),
    elements: Pin<Box<dyn Stream<Item = TCResult<(Coord, Number)>> + Send>>,
}

impl SparseView {
    pub async fn read_from<Txn, FE>(tensor: Sparse<Txn, FE>, txn_id: TxnId) -> TCResult<Self>
    where
        Txn: Transaction<FE>,
        FE: DenseCacheFile + AsType<Node> + Clone,
    {
        let tensor = tensor.into_view();
        let permit = tensor.read_permit(txn_id, Range::default()).await?;

        let schema = (
            ValueType::Number(tensor.dtype()).path(),
            tensor.shape().to_vec(),
        );

        let elements = tensor.into_elements(txn_id).await?;

        Ok(Self {
            schema,
            elements,
            _permit: permit,
        })
    }
}

impl<'en> en::IntoStream<'en> for SparseView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeSeq;

        let mut seq = encoder.encode_seq(Some(2))?;
        seq.encode_element(self.schema)?;
        seq.encode_element(en::SeqStream::from(self.elements))?;
        seq.end()
    }
}

pub enum TensorView {
    Dense(DenseView),
    Sparse(SparseView),
}

impl TensorView {
    pub async fn read_from<Txn, FE>(tensor: Tensor<Txn, FE>, txn_id: TxnId) -> TCResult<Self>
    where
        Txn: Transaction<FE>,
        FE: DenseCacheFile + AsType<Node> + Clone,
    {
        match tensor {
            Tensor::Dense(dense) => {
                DenseView::read_from(dense, txn_id)
                    .map_ok(Self::Dense)
                    .await
            }
            Tensor::Sparse(sparse) => {
                SparseView::read_from(sparse, txn_id)
                    .map_ok(Self::Sparse)
                    .await
            }
        }
    }
}

impl<'en> en::IntoStream<'en> for TensorView {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Dense(dense) => dense.into_stream(encoder),
            Self::Sparse(sparse) => sparse.into_stream(encoder),
        }
    }
}
