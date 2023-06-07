use std::cmp::Ordering;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::{fmt, io};

use async_trait::async_trait;
use destream::de;
use freqfs::{
    DirLock, DirReadGuard, FileLoad, FileReadGuard, FileReadGuardOwned, FileWriteGuardOwned,
};
use futures::future::{Future, FutureExt, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use ha_ndarray::*;
use safecast::AsType;

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::TxnId;
use tc_value::{DType, Float, Int, Number, NumberClass, NumberInstance, NumberType, UInt};

use super::{DenseInstance, DenseWrite, DenseWriteGuard, DenseWriteLock};

use crate::tensor::transform::{Broadcast, Expand, Reduce, Reshape, Slice, Transpose};
use crate::tensor::{
    coord_of, offset_of, Axes, AxisRange, Coord, Range, Semaphore, Shape, TensorInstance,
    TensorPermitRead, TensorPermitWrite, IDEAL_BLOCK_SIZE,
};

use super::stream::BlockResize;
use super::{BlockShape, BlockStream, DenseCacheFile};

pub enum ArrayCastSource {
    F32(Array<f32>),
    F64(Array<f64>),
    I16(Array<i16>),
    I32(Array<i32>),
    I64(Array<i64>),
    U8(Array<u8>),
    U16(Array<u16>),
    U32(Array<u32>),
    U64(Array<u64>),
}

macro_rules! array_cmp {
    ($self:ident, $other:ident, $this:ident, $that:ident, $call:expr) => {
        match ($self, $other) {
            (Self::F32($this), Self::F32($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::F64($this), Self::F64($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::I16($this), Self::I16($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::I32($this), Self::I32($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::I64($this), Self::I64($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U8($this), Self::U8($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U16($this), Self::U16($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U32($this), Self::U32($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            (Self::U64($this), Self::U64($that)) => {
                $this.eq($that).map(Array::from).map_err(TCError::from)
            }
            ($this, $that) => Err(bad_request!("cannot compare {:?} with {:?}", $this, $that)),
        }
    };
}

macro_rules! array_cmp_scalar {
    ($self:ident, $other:ident, $this:ident, $that:ident, $call:expr) => {
        match ($self, $other) {
            (Self::F32($this), Number::Float(Float::F32($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::F64($this), Number::Float(Float::F64($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::I16($this), Number::Int(Int::I16($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::I32($this), Number::Int(Int::I32($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::I64($this), Number::Int(Int::I64($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U8($this), Number::UInt(UInt::U8($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U16($this), Number::UInt(UInt::U16($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U32($this), Number::UInt(UInt::U32($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            (Self::U64($this), Number::UInt(UInt::U64($that))) => $this
                .eq_scalar($that)
                .map(Array::from)
                .map_err(TCError::from),

            ($this, $that) => Err(bad_request!("cannot compare {:?} with {:?}", $this, $that)),
        }
    };
}

impl ArrayCastSource {
    pub fn and(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.and(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn and_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.and_scalar(that)
                .map(Array::from)
                .map_err(TCError::from)
        )
    }

    pub fn not(self) -> TCResult<Array<u8>> {
        match self {
            Self::F32(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::F64(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::I16(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::I32(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::I64(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::U8(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::U16(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::U32(this) => this.not().map(Array::from).map_err(TCError::from),
            Self::U64(this) => this.not().map(Array::from).map_err(TCError::from),
        }
    }

    pub fn or(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.or(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn or_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.or_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn xor(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.xor(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn xor_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.xor_scalar(that)
                .map(Array::from)
                .map_err(TCError::from)
        )
    }

    pub fn eq(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.eq(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn eq_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.eq_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn gt(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.gt(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn gt_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.gt_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ge(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.ge(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ge_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.ge_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn lt(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.lt(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn lt_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.lt_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn le(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.le(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn le_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.le_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ne(self, other: Self) -> TCResult<Array<u8>> {
        array_cmp!(
            self,
            other,
            this,
            that,
            this.ne(that).map(Array::from).map_err(TCError::from)
        )
    }

    pub fn ne_scalar(self, other: Number) -> TCResult<Array<u8>> {
        array_cmp_scalar!(
            self,
            other,
            this,
            that,
            this.ne_scalar(that).map(Array::from).map_err(TCError::from)
        )
    }
}

impl fmt::Debug for ArrayCastSource {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::F32(this) => this.fmt(f),
            Self::F64(this) => this.fmt(f),
            Self::I16(this) => this.fmt(f),
            Self::I32(this) => this.fmt(f),
            Self::I64(this) => this.fmt(f),
            Self::U8(this) => this.fmt(f),
            Self::U16(this) => this.fmt(f),
            Self::U32(this) => this.fmt(f),
            Self::U64(this) => this.fmt(f),
        }
    }
}

pub enum DenseCastSource<FE> {
    F32(DenseAccess<FE, f32>),
    F64(DenseAccess<FE, f64>),
    I16(DenseAccess<FE, i16>),
    I32(DenseAccess<FE, i32>),
    I64(DenseAccess<FE, i64>),
    U8(DenseAccess<FE, u8>),
    U16(DenseAccess<FE, u16>),
    U32(DenseAccess<FE, u32>),
    U64(DenseAccess<FE, u64>),
}

impl<FE> Clone for DenseCastSource<FE> {
    fn clone(&self) -> Self {
        match self {
            Self::F32(access) => Self::F32(access.clone()),
            Self::F64(access) => Self::F64(access.clone()),
            Self::I16(access) => Self::I16(access.clone()),
            Self::I32(access) => Self::I32(access.clone()),
            Self::I64(access) => Self::I64(access.clone()),
            Self::U8(access) => Self::U8(access.clone()),
            Self::U16(access) => Self::U16(access.clone()),
            Self::U32(access) => Self::U32(access.clone()),
            Self::U64(access) => Self::U64(access.clone()),
        }
    }
}

impl<FE> DenseCastSource<FE>
where
    FE: DenseCacheFile,
{
    async fn read_block(&self, block_id: u64) -> TCResult<ArrayCastSource> {
        match self {
            Self::F32(this) => this.read_block(block_id).map_ok(ArrayCastSource::F32).await,
            Self::F64(this) => this.read_block(block_id).map_ok(ArrayCastSource::F64).await,
            Self::I16(this) => this.read_block(block_id).map_ok(ArrayCastSource::I16).await,
            Self::I32(this) => this.read_block(block_id).map_ok(ArrayCastSource::I32).await,
            Self::I64(this) => this.read_block(block_id).map_ok(ArrayCastSource::I64).await,
            Self::U8(this) => this.read_block(block_id).map_ok(ArrayCastSource::U8).await,
            Self::U16(this) => this.read_block(block_id).map_ok(ArrayCastSource::U16).await,
            Self::U32(this) => this.read_block(block_id).map_ok(ArrayCastSource::U32).await,
            Self::U64(this) => this.read_block(block_id).map_ok(ArrayCastSource::U64).await,
        }
    }

    async fn read_blocks(
        self,
    ) -> TCResult<Pin<Box<dyn Stream<Item = TCResult<ArrayCastSource>> + Send>>> {
        match self {
            Self::F32(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::F32))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::F64(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::F64))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::I16(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::I16))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::I32(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::I32))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::I64(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::I64))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::U8(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::U8))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::U16(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::U16))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::U32(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::U32))
                    .await?;

                Ok(Box::pin(blocks))
            }
            Self::U64(this) => {
                let blocks = this
                    .read_blocks()
                    .map_ok(|blocks| blocks.map_ok(ArrayCastSource::U64))
                    .await?;

                Ok(Box::pin(blocks))
            }
        }
    }
}

macro_rules! cast_source {
    ($t:ty, $var:ident) => {
        impl<FE> From<DenseAccess<FE, $t>> for DenseCastSource<FE> {
            fn from(access: DenseAccess<FE, $t>) -> Self {
                Self::$var(access)
            }
        }
    };
}

cast_source!(f32, F32);
cast_source!(f64, F64);
cast_source!(i16, I16);
cast_source!(i32, I32);
cast_source!(i64, I64);
cast_source!(u8, U8);
cast_source!(u16, U16);
cast_source!(u32, U32);
cast_source!(u64, U64);

macro_rules! cast_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            DenseCastSource::F32($var) => $call,
            DenseCastSource::F64($var) => $call,
            DenseCastSource::I16($var) => $call,
            DenseCastSource::I32($var) => $call,
            DenseCastSource::I64($var) => $call,
            DenseCastSource::U8($var) => $call,
            DenseCastSource::U16($var) => $call,
            DenseCastSource::U32($var) => $call,
            DenseCastSource::U64($var) => $call,
        }
    };
}

impl<FE> fmt::Debug for DenseCastSource<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        cast_dispatch!(self, this, this.fmt(f))
    }
}

pub enum DenseAccess<FE, T: CDatatype> {
    File(DenseFile<FE, T>),
    Broadcast(Box<DenseBroadcast<Self>>),
    Cast(Box<DenseCast<FE, T>>),
    Combine(Box<DenseCombine<Self, Self, T>>),
    Compare(Box<DenseCompare<FE, T>>),
    CompareConst(Box<DenseCompareConst<FE, T>>),
    Const(Box<DenseConst<Self, T>>),
    Cow(Box<DenseCow<FE, Self>>),
    Diagonal(Box<DenseDiagonal<Self>>),
    Expand(Box<DenseExpand<Self>>),
    Reduce(Box<DenseReduce<Self, T>>),
    Reshape(Box<DenseReshape<Self>>),
    Slice(Box<DenseSlice<Self>>),
    Transpose(Box<DenseTranspose<Self>>),
    Unary(Box<DenseUnary<Self, T>>),
    UnaryBoolean(Box<DenseUnaryBoolean<FE, T>>),
    Version(DenseVersion<FE, T>),
}

impl<FE, T: CDatatype> Clone for DenseAccess<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::File(file) => Self::File(file.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
            Self::Combine(combine) => Self::Combine(combine.clone()),
            Self::Compare(compare) => Self::Compare(compare.clone()),
            Self::CompareConst(compare) => Self::CompareConst(compare.clone()),
            Self::Const(combine) => Self::Const(combine.clone()),
            Self::Cow(cow) => Self::Cow(cow.clone()),
            Self::Diagonal(diag) => Self::Diagonal(diag.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reduce(reduce) => Self::Reduce(reduce.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
            Self::Unary(unary) => Self::Unary(unary.clone()),
            Self::UnaryBoolean(unary) => Self::UnaryBoolean(unary.clone()),
            Self::Version(version) => Self::Version(version.clone()),
        }
    }
}

macro_rules! access_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::File($var) => $call,
            Self::Broadcast($var) => $call,
            Self::Cast($var) => $call,
            Self::Combine($var) => $call,
            Self::Compare($var) => $call,
            Self::CompareConst($var) => $call,
            Self::Const($var) => $call,
            Self::Cow($var) => $call,
            Self::Diagonal($var) => $call,
            Self::Expand($var) => $call,
            Self::Reduce($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
            Self::Transpose($var) => $call,
            Self::Unary($var) => $call,
            Self::UnaryBoolean($var) => $call,
            Self::Version($var) => $call,
        }
    };
}

impl<FE, T> TensorInstance for DenseAccess<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        access_dispatch!(self, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseAccess<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>>,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        access_dispatch!(self, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        access_dispatch!(
            self,
            this,
            this.read_block(block_id).map_ok(Array::from).await
        )
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        match self {
            Self::Cast(cast) => cast.read_blocks().await,
            Self::File(file) => Ok(Box::pin(file.read_blocks().await?.map_ok(Array::from))),
            Self::Broadcast(broadcast) => {
                Ok(Box::pin(broadcast.read_blocks().await?.map_ok(Array::from)))
            }
            Self::Combine(combine) => combine.read_blocks().await,
            Self::Compare(compare) => compare.read_blocks().await,
            Self::CompareConst(compare) => compare.read_blocks().await,
            Self::Const(combine) => combine.read_blocks().await,
            Self::Cow(cow) => cow.read_blocks().await,
            Self::Diagonal(diag) => Ok(Box::pin(diag.read_blocks().await?.map_ok(Array::from))),
            Self::Expand(expand) => Ok(Box::pin(expand.read_blocks().await?.map_ok(Array::from))),
            Self::Reduce(reduce) => reduce.read_blocks().await,
            Self::Reshape(reshape) => {
                Ok(Box::pin(reshape.read_blocks().await?.map_ok(Array::from)))
            }
            Self::Slice(slice) => Ok(Box::pin(slice.read_blocks().await?.map_ok(Array::from))),
            Self::Transpose(transpose) => {
                Ok(Box::pin(transpose.read_blocks().await?.map_ok(Array::from)))
            }
            Self::Unary(unary) => unary.read_blocks().await,
            Self::UnaryBoolean(unary) => unary.read_blocks().await,
            Self::Version(version) => {
                Ok(Box::pin(version.read_blocks().await?.map_ok(Array::from)))
            }
        }
    }
}

#[async_trait]
impl<FE, T> TensorPermitRead for DenseAccess<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::Cast(cast) => cast.read_permit(txn_id, range).await,
            Self::Combine(combine) => combine.read_permit(txn_id, range).await,
            Self::Const(combine) => combine.read_permit(txn_id, range).await,
            Self::Cow(cow) => cow.read_permit(txn_id, range).await,
            Self::Diagonal(diag) => diag.read_permit(txn_id, range).await,
            Self::Expand(expand) => expand.read_permit(txn_id, range).await,
            Self::Reshape(reshape) => reshape.read_permit(txn_id, range).await,
            Self::Slice(slice) => slice.read_permit(txn_id, range).await,
            Self::Transpose(transpose) => transpose.read_permit(txn_id, range).await,
            Self::Version(version) => version.read_permit(txn_id, range).await,

            other => Err(bad_request!(
                "{:?} does not support transactional locking",
                other
            )),
        }
    }
}

#[async_trait]
impl<FE, T> TensorPermitWrite for DenseAccess<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        match self {
            Self::Slice(slice) => slice.write_permit(txn_id, range).await,
            Self::Version(version) => version.write_permit(txn_id, range).await,
            other => Err(bad_request!(
                "{:?} does not support transactional writes",
                other
            )),
        }
    }
}

impl<FE, T: CDatatype + DType> fmt::Debug for DenseAccess<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        access_dispatch!(self, this, this.fmt(f))
    }
}

pub struct DenseFile<FE, T> {
    dir: DirLock<FE>,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
    shape: Shape,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for DenseFile<FE, T> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            block_map: self.block_map.clone(),
            block_size: self.block_size,
            shape: self.shape.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE, T> DenseFile<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>>,
    T: CDatatype + DType + NumberInstance,
    Buffer<T>: de::FromStream<Context = ()>,
{
    pub async fn load(dir: DirLock<FE>, shape: Shape) -> TCResult<Self> {
        let contents = dir.read().await;
        let num_blocks = contents.len();

        if num_blocks == 0 {
            return Err(bad_request!(
                "cannot load a dense tensor from an empty directory"
            ));
        }

        let mut size = 0u64;

        let block_size = {
            let block = contents
                .get_file(&0)
                .ok_or_else(|| TCError::not_found("block 0"))?;

            let block: FileReadGuard<Buffer<T>> = block.read().await?;
            size += block.len() as u64;
            block.len()
        };

        let block_axis = block_axis_for(&shape, block_size);
        let block_shape = block_shape_for(block_axis, &shape, block_size);

        for block_id in 1..(num_blocks - 1) {
            let block = contents
                .get_file(&block_id)
                .ok_or_else(|| TCError::not_found(format!("block {block_id}")))?;

            let block: FileReadGuard<Buffer<T>> = block.read().await?;
            if block.len() == block_size {
                size += block.len() as u64;
            } else {
                return Err(bad_request!(
                    "block {} has incorrect size {} (expected {})",
                    block_id,
                    block.len(),
                    block_size
                ));
            }
        }

        {
            let block_id = num_blocks - 1;
            let block = contents
                .get_file(&block_id)
                .ok_or_else(|| bad_request!("block {block_id}"))?;

            let block: FileReadGuard<Buffer<T>> = block.read().await?;
            size += block.len() as u64;
        }

        std::mem::drop(contents);

        if size != shape.size() {
            return Err(bad_request!(
                "tensor blocks have incorrect total length {} (expected {} for shape {:?})",
                size,
                shape.size(),
                shape
            ));
        }

        let mut block_map_shape = BlockShape::with_capacity(block_axis + 1);
        block_map_shape.extend(
            shape
                .iter()
                .take(block_axis)
                .copied()
                .map(|dim| dim as usize),
        );
        block_map_shape.push(shape[block_axis] as usize / block_shape[0]);

        let block_map = ArrayBase::<Vec<_>>::new(
            block_map_shape,
            (0..num_blocks as u64).into_iter().collect(),
        )?;

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }

    pub async fn constant(dir: DirLock<FE>, shape: Shape, value: T) -> TCResult<Self> {
        shape.validate()?;

        let ndim = shape.len();
        let size = shape.iter().product();

        let ideal_block_size = IDEAL_BLOCK_SIZE as u64;
        let (block_size, num_blocks) = if size < (2 * ideal_block_size) {
            (size as usize, 1)
        } else if ndim == 1 && size % ideal_block_size == 0 {
            (IDEAL_BLOCK_SIZE, (size / ideal_block_size) as usize)
        } else if ndim == 1
            || (shape.iter().rev().take(2).product::<u64>() > (2 * ideal_block_size))
        {
            let num_blocks = div_ceil(size, ideal_block_size) as usize;
            (IDEAL_BLOCK_SIZE, num_blocks as usize)
        } else {
            let matrix_size = shape.iter().rev().take(2).product::<u64>();
            let block_size = ideal_block_size + (matrix_size - (ideal_block_size % matrix_size));
            let num_blocks = div_ceil(size, ideal_block_size);
            (block_size as usize, num_blocks as usize)
        };

        debug_assert!(block_size > 0);

        {
            let dtype_size = T::dtype().size();

            let mut dir = dir.write().await;

            for block_id in 0..num_blocks {
                dir.create_file::<Buffer<T>>(
                    block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )?;
            }

            let last_block_id = num_blocks - 1;
            if size % block_size as u64 == 0 {
                dir.create_file::<Buffer<T>>(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )
            } else {
                dir.create_file::<Buffer<T>>(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )
            }?;
        };

        let block_axis = block_axis_for(&shape, block_size);
        let map_shape = shape
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect();

        let block_map =
            ArrayBase::<Vec<_>>::new(map_shape, (0u64..num_blocks as u64).into_iter().collect())?;

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for DenseFile<FE, T>
where
    FE: Send + Sync + 'static,
    T: DType + Send + Sync + 'static,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = ArrayBase<FileReadGuardOwned<FE, Buffer<T>>>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let dir = self.dir.read().await;
        let file = dir.get_file(&block_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("dense tensor block {}", block_id),
            )
        })?;

        let buffer = file.read_owned().await?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, buffer.len());
        ArrayBase::<FileReadGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
            .map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let shape = self.shape;
        let block_axis = block_axis_for(&shape, self.block_size);
        let dir = self.dir.into_read().await;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                dir.get_file(&block_id).cloned().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("dense tensor block {}", block_id),
                    )
                    .into()
                })
            })
            .map_ok(|block| block.into_read())
            .try_buffered(num_cpus::get())
            .map(move |result| {
                let buffer = result?;
                let block_shape = block_shape_for(block_axis, &shape, buffer.len());
                ArrayBase::<FileReadGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<'a, FE, T> DenseWrite for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type BlockWrite = ArrayBase<FileWriteGuardOwned<FE, Buffer<T>>>;

    async fn write_block(&self, block_id: u64) -> TCResult<Self::BlockWrite> {
        let dir = self.dir.read().await;
        let file = dir.get_file(&block_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("dense tensor block {}", block_id),
            )
        })?;

        let buffer = file.write_owned().await?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, buffer.len());
        ArrayBase::<FileWriteGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
            .map_err(TCError::from)
    }

    async fn write_blocks(self) -> TCResult<BlockStream<Self::BlockWrite>> {
        let shape = self.shape;
        let block_axis = block_axis_for(&shape, self.block_size);
        let dir = self.dir.into_read().await;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                dir.get_file(&block_id).cloned().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("dense tensor block {}", block_id),
                    )
                    .into()
                })
            })
            .map_ok(|block| block.into_write())
            .try_buffered(num_cpus::get())
            .map(move |result| {
                let buffer = result?;
                let block_shape = block_shape_for(block_axis, &shape, buffer.len());
                ArrayBase::<FileWriteGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<'a, FE, T> DenseWriteLock<'a> for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type WriteGuard = DenseFileWriteGuard<'a, FE>;

    async fn write(&'a self) -> Self::WriteGuard {
        let dir = self.dir.read().await;

        DenseFileWriteGuard {
            dir: Arc::new(dir),
            block_size: self.block_size,
            shape: &self.shape,
        }
    }
}

impl<FE, T: CDatatype> From<DenseFile<FE, T>> for DenseAccess<FE, T> {
    fn from(file: DenseFile<FE, T>) -> Self {
        Self::File(file)
    }
}

impl<FE, T> fmt::Debug for DenseFile<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dense tensor with shape {:?}", self.shape)
    }
}

pub struct DenseFileWriteGuard<'a, FE> {
    dir: Arc<DirReadGuard<'a, FE>>,
    block_size: usize,
    shape: &'a Shape,
}

impl<'a, FE> DenseFileWriteGuard<'a, FE> {
    pub async fn merge<T>(&self, other: DirLock<FE>) -> TCResult<()>
    where
        FE: FileLoad + AsType<Buffer<T>>,
        T: CDatatype + DType + 'static,
        Buffer<T>: de::FromStream<Context = ()>,
    {
        let num_blocks = div_ceil(self.shape.size(), self.block_size as u64);
        stream::iter(0..num_blocks)
            .map(move |block_id| {
                let that = other.clone();

                async move {
                    let that = that.read().await;
                    if that.contains(&block_id) {
                        let mut this = self.dir.write_file(&block_id).await?;
                        let that = that.read_file(&block_id).await?;
                        this.write(&*that).map_err(TCError::from)
                    } else {
                        Ok(())
                    }
                }
            })
            .buffer_unordered(num_cpus::get())
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }
}

#[async_trait]
impl<'a, FE, T> DenseWriteGuard<T> for DenseFileWriteGuard<'a, FE>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    async fn overwrite<O: DenseInstance<DType = T>>(&self, other: O) -> TCResult<()> {
        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, block_shape.iter().product())?;

        let blocks = other.read_blocks().await?;
        let blocks = BlockResize::new(blocks, block_shape)?;

        blocks
            .enumerate()
            .map(|(block_id, result)| {
                let dir = self.dir.clone();
                let queue = queue.clone();

                async move {
                    let data = result?;
                    let data = data.read(&queue)?;
                    let mut block = dir.write_file(&block_id).await?;
                    debug_assert_eq!(block.len(), data.len());
                    block.write(data)?;
                    Result::<(), TCError>::Ok(())
                }
            })
            .buffered(num_cpus::get())
            .try_fold((), |(), ()| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, value: T) -> TCResult<()> {
        let num_blocks = div_ceil(self.shape.size(), self.block_size as u64);

        stream::iter(0..num_blocks)
            .map(|block_id| async move {
                let mut block = self.dir.write_file(&block_id).await?;
                block.write_value(value).map_err(TCError::from)
            })
            .buffered(num_cpus::get())
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, coord: Coord, value: T) -> TCResult<()> {
        self.shape.validate_coord(&coord)?;

        let offset = offset_of(coord, &self.shape);
        let block_id = offset / self.block_size as u64;

        let mut block = self.dir.write_file(&block_id).await?;
        block.write_value_at((offset % self.block_size as u64) as usize, value)?;

        Ok(())
    }
}

pub struct DenseVersion<FE, T> {
    file: DenseFile<FE, T>,
    semaphore: Semaphore,
}

impl<FE, T> Clone for DenseVersion<FE, T> {
    fn clone(&self) -> Self {
        Self {
            file: self.file.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}

impl<FE, T> DenseVersion<FE, T> {
    pub fn new(file: DenseFile<FE, T>, semaphore: Semaphore) -> Self {
        Self { file, semaphore }
    }

    pub fn commit(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, false)
    }

    pub fn rollback(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, false)
    }

    pub fn finalize(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, true)
    }
}

impl<FE, T> TensorInstance for DenseVersion<FE, T>
where
    DenseFile<FE, T>: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.file.dtype()
    }

    fn shape(&self) -> &Shape {
        self.file.shape()
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
    DenseFile<FE, T>: DenseInstance,
{
    type Block = <DenseFile<FE, T> as DenseInstance>::Block;
    type DType = <DenseFile<FE, T> as DenseInstance>::DType;

    fn block_size(&self) -> usize {
        self.file.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.file.read_block(block_id).await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        self.file.read_blocks().await
    }
}

#[async_trait]
impl<FE, T> TensorPermitRead for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.semaphore
            .read(txn_id, range)
            .map_ok(|permit| vec![permit])
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<FE, T> TensorPermitWrite for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.semaphore
            .write(txn_id, range)
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<'a, FE, T> DenseWriteLock<'a> for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
    DenseFile<FE, T>: DenseWriteLock<'a>,
{
    type WriteGuard = <DenseFile<FE, T> as DenseWriteLock<'a>>::WriteGuard;

    async fn write(&'a self) -> Self::WriteGuard {
        self.file.write().await
    }
}

impl<FE, T: CDatatype> From<DenseVersion<FE, T>> for DenseAccess<FE, T> {
    fn from(version: DenseVersion<FE, T>) -> Self {
        Self::Version(version)
    }
}

impl<FE, T> fmt::Debug for DenseVersion<FE, T>
where
    DenseFile<FE, T>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional version of {:?}", self.file)
    }
}

#[derive(Clone)]
pub struct DenseBroadcast<S> {
    source: S,
    transform: Broadcast,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseBroadcast<S> {
    pub fn new(source: S, shape: Shape) -> TCResult<Self> {
        let transform = Broadcast::new(source.shape().clone(), shape)?;

        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());
        let source_block_shape = block_shape_for(block_axis, source.shape(), source.block_size());

        let mut block_shape = BlockShape::with_capacity(source_block_shape.len());
        block_shape.push(source_block_shape[0]);
        block_shape.extend(
            transform
                .shape()
                .iter()
                .rev()
                .take(source_block_shape.len() - 1)
                .rev()
                .copied()
                .map(|dim| dim as usize),
        );

        let block_size = block_shape.iter().product();

        let mut block_map_shape = BlockShape::with_capacity(source.ndim());
        block_map_shape.extend(
            transform
                .shape()
                .iter()
                .take(block_axis)
                .copied()
                .map(|dim| dim as usize),
        );
        block_map_shape.push(transform.shape()[block_axis] as usize / source_block_shape[0]);

        let block_map =
            ArrayBase::<Vec<_>>::new(block_map_shape, (0..num_blocks).into_iter().collect())?;

        Ok(Self {
            source,
            transform,
            block_map,
            block_size,
        })
    }
}

impl<S: TensorInstance> TensorInstance for DenseBroadcast<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseBroadcast<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Broadcast:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Broadcast;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);
        let source_block = self.source.read_block(source_block_id).await?;
        source_block.broadcast(block_shape).map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(block_id).await }
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let source_block = result?;
                source_block
                    .broadcast(block_shape.to_vec())
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseBroadcast<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<FE, T: CDatatype, S: Into<DenseAccess<FE, T>>> From<DenseBroadcast<S>> for DenseAccess<FE, T> {
    fn from(broadcast: DenseBroadcast<S>) -> Self {
        Self::Broadcast(Box::new(DenseBroadcast {
            source: broadcast.source.into(),
            transform: broadcast.transform.clone(),
            block_map: broadcast.block_map,
            block_size: broadcast.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseBroadcast<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "broadcast of {:?} into {:?}",
            self.source,
            self.transform.shape()
        )
    }
}

pub struct DenseCast<FE, T> {
    source: DenseCastSource<FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for DenseCast<FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            dtype: self.dtype,
        }
    }
}

impl<FE, T> DenseCast<FE, T> {
    pub fn new<S: Into<DenseCastSource<FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            dtype: PhantomData,
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for DenseCast<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let source = &self.source;
        cast_dispatch!(source, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseCast<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>>,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let source = &self.source;
        cast_dispatch!(source, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let source = &self.source;

        cast_dispatch!(
            source,
            this,
            this.read_block(block_id)
                .map(|result| result
                    .and_then(|block| block.cast().map(Array::from).map_err(TCError::from)))
                .await
        )
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let source = self.source;

        cast_dispatch!(source, this, {
            let source_blocks = this.read_blocks().await?;
            let blocks = source_blocks.map(|result| {
                result.and_then(|block| block.cast().map(Array::from).map_err(TCError::from))
            });

            Ok(Box::pin(blocks))
        })
    }
}

#[async_trait]
impl<FE, T> TensorPermitRead for DenseCast<FE, T>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
    DenseAccess<FE, T>: TensorPermitRead,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        let source = &self.source;
        cast_dispatch!(source, this, this.read_permit(txn_id, range).await)
    }
}

impl<FE, T: DType> fmt::Debug for DenseCast<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cast {:?} into {:?}", self.source, T::dtype())
    }
}

pub struct DenseCow<FE, S> {
    source: S,
    dir: DirLock<FE>,
}

impl<FE, S: Clone> Clone for DenseCow<FE, S> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            dir: self.dir.clone(),
        }
    }
}

impl<FE, S> DenseCow<FE, S> {
    pub fn create(source: S, dir: DirLock<FE>) -> Self {
        Self { source, dir }
    }
}

impl<FE, S> DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    async fn write_buffer(
        &self,
        block_id: u64,
    ) -> TCResult<FileWriteGuardOwned<FE, Buffer<S::DType>>> {
        let mut dir = self.dir.write().await;

        if let Some(buffer) = dir.get_file(&block_id) {
            buffer.write_owned().map_err(TCError::from).await
        } else {
            let block = self.source.read_block(block_id).await?;

            let context = ha_ndarray::Context::default()?;
            let queue = ha_ndarray::Queue::new(context, block.size())?;
            let buffer = block.read(&queue)?.into_buffer()?;

            let type_size = S::DType::dtype().size();
            let buffer_data_size = type_size * buffer.len();
            let buffer = dir.create_file(block_id.to_string(), buffer, buffer_data_size)?;

            buffer.into_write().map_err(TCError::from).await
        }
    }
}

impl<FE, S> TensorInstance for DenseCow<FE, S>
where
    FE: Send + Sync + 'static,
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<FE, S> DenseInstance for DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    type Block = Array<S::DType>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let dir = self.dir.read().await;

        if let Some(block) = dir.get_file(&block_id) {
            let buffer: Buffer<S::DType> = block
                .read_owned::<Buffer<S::DType>>()
                .map_ok(|block| block.clone().into())
                .map_err(TCError::from)
                .await?;

            let block_axis = block_axis_for(self.shape(), self.block_size());
            let block_data_size = S::DType::dtype().size() * buffer.len();
            let block_shape = block_shape_for(block_axis, self.shape(), block_data_size);
            let block = ArrayBase::<Buffer<S::DType>>::new(block_shape, buffer)?;

            Ok(block.into())
        } else {
            self.source.read_block(block_id).map_ok(Array::from).await
        }
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);

        let blocks = stream::iter(0..num_blocks)
            .map(move |block_id| {
                let this = self.clone();
                async move { this.read_block(block_id).await }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<FE, S> TensorPermitRead for DenseCow<FE, S>
where
    FE: Send + Sync,
    S: TensorPermitRead,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.source.read_permit(txn_id, range).await
    }
}

#[async_trait]
impl<FE, S> TensorPermitWrite for DenseCow<FE, S>
where
    FE: Send + Sync,
    S: TensorPermitWrite,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.source.write_permit(txn_id, range).await
    }
}

#[async_trait]
impl<FE, S> DenseWrite for DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    type BlockWrite = ArrayBase<FileWriteGuardOwned<FE, Buffer<S::DType>>>;

    async fn write_block(&self, block_id: u64) -> TCResult<Self::BlockWrite> {
        let buffer = self.write_buffer(block_id).await?;
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let block_shape = block_shape_for(block_axis, self.shape(), buffer.len());
        ArrayBase::<FileWriteGuardOwned<FE, Buffer<S::DType>>>::new(block_shape, buffer)
            .map_err(TCError::from)
    }

    async fn write_blocks(self) -> TCResult<BlockStream<Self::BlockWrite>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);
        let blocks = stream::iter(0..num_blocks).then(move |block_id| {
            let this = self.clone();
            async move { this.write_block(block_id).await }
        });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<'a, FE, S> DenseWriteLock<'a> for DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    type WriteGuard = DenseCowWriteGuard<'a, FE, S>;

    async fn write(&'a self) -> Self::WriteGuard {
        DenseCowWriteGuard { cow: self }
    }
}

impl<'a, FE, S, T> From<DenseCow<FE, S>> for DenseAccess<FE, T>
where
    T: CDatatype,
    DenseAccess<FE, T>: From<S>,
{
    fn from(cow: DenseCow<FE, S>) -> Self {
        Self::Cow(Box::new(DenseCow {
            source: cow.source.into(),
            dir: cow.dir,
        }))
    }
}

impl<FE, S: fmt::Debug> fmt::Debug for DenseCow<FE, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "copy-on-write view of {:?}", self.source)
    }
}

pub struct DenseCowWriteGuard<'a, FE, S> {
    cow: &'a DenseCow<FE, S>,
}

#[async_trait]
impl<'a, FE, S> DenseWriteGuard<S::DType> for DenseCowWriteGuard<'a, FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    async fn overwrite<O: DenseInstance<DType = S::DType>>(&self, other: O) -> TCResult<()> {
        let source = other.read_blocks().await?;

        let block_axis = block_axis_for(self.cow.shape(), self.cow.block_size());
        let block_shape = block_shape_for(block_axis, self.cow.shape(), self.cow.block_size());
        let source = BlockResize::new(source, block_shape)?;

        let dest = self.cow.clone().write_blocks().await?;

        dest.zip(source)
            .map(|(dest, source)| {
                let mut dest = dest?;
                let source = source?;
                dest.write(&source).map_err(TCError::from)
            })
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, value: S::DType) -> TCResult<()> {
        let dest = self.cow.clone().write_blocks().await?;
        dest.map_ok(|mut block| block.write_value(value))
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, coord: Coord, value: S::DType) -> TCResult<()> {
        self.cow.shape().validate_coord(&coord)?;

        let offset = offset_of(coord, self.cow.shape());
        let block_id = offset / self.cow.block_size() as u64;
        let block_offset = offset % self.cow.block_size() as u64;
        let mut buffer = self.cow.write_buffer(block_id).await?;

        buffer
            .write_value_at(block_offset as usize, value)
            .map_err(TCError::from)
    }
}

#[derive(Clone)]
pub struct DenseDiagonal<S> {
    source: S,
    shape: Shape,
}

impl<S: TensorInstance + fmt::Debug> DenseDiagonal<S> {
    pub fn new(source: S) -> TCResult<Self> {
        if source.shape().len() >= 2
            && source.shape().iter().nth_back(0) == source.shape().iter().nth_back(1)
        {
            let mut shape = source.shape().to_vec();
            shape.pop();

            Ok(Self {
                source,
                shape: shape.into(),
            })
        } else {
            Err(bad_request!(
                "matrix diagonal requires a square matrix or batch of square matrices, not {:?}",
                source
            ))
        }
    }
}

impl<S: TensorInstance> TensorInstance for DenseDiagonal<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseDiagonal<S> {
    type Block = ArrayOp<ha_ndarray::ops::MatDiag<S::Block>>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        let matrix_dim = self.shape.last().copied().expect("matrix dim") as usize;
        self.source.block_size() / matrix_dim
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.source
            .read_block(block_id)
            .map(|result| result.and_then(|block| block.diagonal().map_err(TCError::from)))
            .await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let source_blocks = self.source.read_blocks().await?;
        let blocks = source_blocks
            .map(|result| result.and_then(|block| block.diagonal().map_err(TCError::from)));

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: TensorInstance + TensorPermitRead + fmt::Debug> TensorPermitRead for DenseDiagonal<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        mut range: Range,
    ) -> TCResult<Vec<PermitRead<Range>>> {
        let range = match range.len().cmp(&self.ndim()) {
            Ordering::Less => Ok(range),
            Ordering::Equal => {
                let axis_range = range.last().cloned().expect("last axis range");
                range.push(axis_range);
                Ok(range)
            }
            Ordering::Greater => Err(bad_request!("invalid range for {:?}: {:?}", self, range)),
        }?;

        self.source.read_permit(txn_id, range).await
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseDiagonal<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "matrix diagonal of {:?}", self.source)
    }
}

#[derive(Clone)]
pub struct DenseExpand<S> {
    source: S,
    transform: Expand,
}

impl<S: DenseInstance> DenseExpand<S> {
    pub fn new(source: S, axes: Axes) -> TCResult<Self> {
        Expand::new(source.shape().clone(), axes).map(|transform| Self { source, transform })
    }
}

impl<S: TensorInstance> TensorInstance for DenseExpand<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseExpand<S> {
    type Block = S::Block;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.source.read_block(block_id).await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        self.source.read_blocks().await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for DenseExpand<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseExpand<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "expand axes {:?} of {:?}",
            self.transform.expand_axes(),
            self.source,
        )
    }
}

type Combine<T> = fn(Array<T>, Array<T>) -> TCResult<Array<T>>;

#[derive(Clone)]
pub struct DenseCombine<L, R, T: CDatatype> {
    left: L,
    right: R,
    op: Combine<T>,
}

impl<L, R, T> DenseCombine<L, R, T>
where
    L: DenseInstance + fmt::Debug,
    R: DenseInstance + fmt::Debug,
    T: CDatatype + DType,
{
    pub fn new(left: L, right: R, op: Combine<T>) -> TCResult<Self> {
        if left.block_size() == right.block_size() && left.shape() == right.shape() {
            Ok(Self { left, right, op })
        } else {
            Err(bad_request!("cannot combine {:?} with {:?}", left, right))
        }
    }
}

impl<L, R, T> TensorInstance for DenseCombine<L, R, T>
where
    L: TensorInstance,
    R: TensorInstance,
    T: DType + CDatatype,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.left.shape(), self.right.shape());
        self.left.shape()
    }
}

#[async_trait]
impl<L, R, T> DenseInstance for DenseCombine<L, R, T>
where
    L: DenseInstance<DType = T>,
    R: DenseInstance<DType = T>,
    T: CDatatype + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.left.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let (left, right) = try_join!(
            self.left.read_block(block_id),
            self.right.read_block(block_id)
        )?;

        (self.op)(left.into(), right.into())
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let op = self.op;

        let (left, right) = try_join!(self.left.read_blocks(), self.right.read_blocks())?;

        let blocks = left.zip(right).map(move |(l, r)| {
            let l = l?;
            let r = r?;
            (op)(l.into(), r.into())
        });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<L, R, T> TensorPermitRead for DenseCombine<L, R, T>
where
    L: TensorPermitRead,
    R: TensorPermitRead,
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these locks in the same order, to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<L, R, T> fmt::Debug for DenseCombine<L, R, T>
where
    L: fmt::Debug,
    R: fmt::Debug,
    T: CDatatype,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with {:?}", self.left, self.right)
    }
}

type ArrayCmp<T> = fn(ArrayCastSource, ArrayCastSource) -> TCResult<Array<T>>;

pub struct DenseCompare<FE, T: CDatatype> {
    left: DenseCastSource<FE>,
    right: DenseCastSource<FE>,
    cmp: ArrayCmp<T>,
}

impl<FE, T: CDatatype> Clone for DenseCompare<FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            cmp: self.cmp,
        }
    }
}

impl<FE, T: CDatatype> DenseCompare<FE, T> {
    pub fn new<L, R>(left: L, right: R, cmp: ArrayCmp<T>) -> TCResult<Self>
    where
        L: DenseInstance + Into<DenseCastSource<FE>> + fmt::Debug,
        R: DenseInstance + Into<DenseCastSource<FE>> + fmt::Debug,
    {
        if left.block_size() == right.block_size() && left.shape() == right.shape() {
            Ok(Self {
                left: left.into(),
                right: right.into(),
                cmp,
            })
        } else {
            Err(bad_request!("cannot compare {:?} with {:?}", left, right))
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for DenseCompare<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let left = &self.left;
        cast_dispatch!(left, this, this.shape())
    }
}

#[async_trait]
impl<FE: DenseCacheFile, T: CDatatype + DType> DenseInstance for DenseCompare<FE, T> {
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let left = &self.left;
        cast_dispatch!(left, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let (left, right) = try_join!(
            self.left.read_block(block_id),
            self.right.read_block(block_id)
        )?;

        (self.cmp)(left, right)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let (left, right) = try_join!(self.left.read_blocks(), self.right.read_blocks())?;

        let blocks = left.zip(right).map(move |(l, r)| {
            let (l, r) = (l?, r?);
            (self.cmp)(l, r)
        });

        Ok(Box::pin(blocks))
    }
}

impl<FE, T: CDatatype> fmt::Debug for DenseCompare<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "compare {:?} with {:?}", self.left, self.right)
    }
}

type ArrayCmpScalar<T> = fn(ArrayCastSource, Number) -> TCResult<Array<T>>;

pub struct DenseCompareConst<FE, T: CDatatype> {
    left: DenseCastSource<FE>,
    right: Number,
    cmp: ArrayCmpScalar<T>,
}

impl<FE, T: CDatatype> Clone for DenseCompareConst<FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right,
            cmp: self.cmp,
        }
    }
}

impl<FE, T: CDatatype> DenseCompareConst<FE, T> {
    pub fn new<L, R>(left: L, right: R, cmp: ArrayCmpScalar<T>) -> Self
    where
        L: Into<DenseCastSource<FE>>,
        R: Into<Number>,
    {
        Self {
            left: left.into(),
            right: right.into(),
            cmp,
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for DenseCompareConst<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let left = &self.left;
        cast_dispatch!(left, this, this.shape())
    }
}

#[async_trait]
impl<FE: DenseCacheFile, T: CDatatype + DType> DenseInstance for DenseCompareConst<FE, T> {
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let left = &self.left;
        cast_dispatch!(left, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.left
            .read_block(block_id)
            .map(|result| result.and_then(move |block| (self.cmp)(block, self.right)))
            .await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let left = self.left.read_blocks().await?;
        let blocks = left.map(move |result| result.and_then(|block| (self.cmp)(block, self.right)));
        Ok(Box::pin(blocks))
    }
}

impl<FE, T: CDatatype> fmt::Debug for DenseCompareConst<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "compare {:?} with {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct DenseConst<L, T: CDatatype> {
    left: L,
    right: T,
    op: fn(Array<T>, T) -> TCResult<Array<T>>,
}

impl<L, T: CDatatype> DenseConst<L, T> {
    pub fn new(left: L, right: T, op: fn(Array<T>, T) -> TCResult<Array<T>>) -> Self {
        Self { left, right, op }
    }
}

impl<L, T> TensorInstance for DenseConst<L, T>
where
    L: TensorInstance,
    T: CDatatype + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.left.shape()
    }
}

#[async_trait]
impl<L, T> DenseInstance for DenseConst<L, T>
where
    L: DenseInstance<DType = T> + fmt::Debug,
    T: CDatatype + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.left.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.left
            .read_block(block_id)
            .map(move |result| result.and_then(|block| (self.op)(block.into(), self.right)))
            .await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let left = self.left.read_blocks().await?;
        let blocks =
            left.map(move |result| result.and_then(|block| (self.op)(block.into(), self.right)));

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<L: TensorPermitRead, T: CDatatype> TensorPermitRead for DenseConst<L, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<L: fmt::Debug, T: CDatatype> fmt::Debug for DenseConst<L, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dual constant operation on {:?}", self.left)
    }
}

#[derive(Clone)]
pub struct DenseReduce<S, T: CDatatype> {
    source: S,
    transform: Reduce,
    block_map: ArrayBase<Arc<Vec<u64>>>,
    map_axes: Axes,
    block_axes: Axes,
    reduce_blocks: Combine<T>,
    reduce_op: fn(Array<T>, &[usize], bool) -> TCResult<Array<T>>,
}

impl<S: DenseInstance> DenseReduce<S, S::DType>
where
    Array<S::DType>: Clone,
{
    fn new(
        source: S,
        axes: Axes,
        keepdims: bool,
        reduce_blocks: Combine<S::DType>,
        reduce_op: fn(Array<S::DType>, &[usize], bool) -> TCResult<Array<S::DType>>,
    ) -> TCResult<Self> {
        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());
        let block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        debug_assert_eq!(source.shape()[block_axis] % block_shape[0] as u64, 0);

        let map_axes = axes[..(block_axis + 1)].to_vec();
        let block_axes = axes[block_axis..].to_vec();

        let transform = Reduce::new(source.shape().clone(), axes, keepdims)?;

        let mut block_map_shape = Vec::with_capacity(source.ndim());
        block_map_shape.extend(
            source.shape()[..block_axis]
                .iter()
                .copied()
                .map(|dim| dim as usize),
        );
        block_map_shape.push(source.shape()[block_axis] as usize / block_shape[0]);

        let block_map = (0..num_blocks).into_iter().collect();
        let block_map = ArrayBase::<Arc<Vec<u64>>>::new(block_map_shape, Arc::new(block_map))?;

        Ok(Self {
            source,
            transform,
            block_map,
            map_axes,
            block_axes,
            reduce_blocks,
            reduce_op,
        })
    }

    pub fn max(source: S, axes: Axes, keepdims: bool) -> TCResult<Self> {
        Self::new(
            source,
            axes,
            keepdims,
            |l, r| {
                l.clone()
                    .ge(r.clone())?
                    .cond(l, r)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.max_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
            },
        )
    }

    pub fn min(source: S, axes: Axes, keepdims: bool) -> TCResult<Self> {
        Self::new(
            source,
            axes,
            keepdims,
            |l, r| {
                l.clone()
                    .le(r.clone())?
                    .cond(l, r)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.min_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
            },
        )
    }

    pub fn sum(source: S, axes: Axes, keepdims: bool) -> TCResult<Self> {
        Self::new(
            source,
            axes,
            keepdims,
            |l, r| l.add(r).map(Array::from).map_err(TCError::from),
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.sum_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
            },
        )
    }

    pub fn product(source: S, axes: Axes, keepdims: bool) -> TCResult<Self> {
        Self::new(
            source,
            axes,
            keepdims,
            |l, r| l.mul(r).map(Array::from).map_err(TCError::from),
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.product_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
            },
        )
    }
}

impl<S: TensorInstance, T: CDatatype> TensorInstance for DenseReduce<S, T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S, T> DenseInstance for DenseReduce<S, T>
where
    S: DenseInstance<Block = Array<T>, DType = T> + Clone,
    T: CDatatype + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let block_axis = block_axis_for(self.source.shape(), self.source.block_size());
        let block_shape =
            block_shape_for(block_axis, self.source.shape(), self.source.block_size());

        block_shape
            .iter()
            .enumerate()
            .filter_map(|(x, dim)| {
                if self.block_axes.contains(&x) {
                    None
                } else {
                    Some(dim)
                }
            })
            .product()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let source_blocks_per_block = self
            .map_axes
            .iter()
            .copied()
            .map(|x| self.block_map.shape()[x])
            .product::<usize>();

        let source_block_id = block_id * source_blocks_per_block as u64;
        let map_strides = ha_ndarray::strides_for(self.block_map.shape(), self.block_map.ndim());
        let map_coord = coord_of(
            source_block_id as usize,
            &map_strides,
            self.block_map.shape(),
        );

        let mut map_slice = map_coord
            .into_iter()
            .map(|i| ha_ndarray::AxisBound::At(i))
            .collect::<Vec<_>>();

        for x in self.map_axes.iter().copied() {
            map_slice[x] = ha_ndarray::AxisBound::In(0, self.block_map.shape()[x], 1);
        }

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, self.block_map.size())?;
        let block_map_slice = self.block_map.clone().slice(map_slice)?;
        let source_blocks = block_map_slice.read(&queue)?.to_slice()?;

        debug_assert_eq!(source_blocks.as_ref()[0], source_block_id);

        let block = self
            .source
            .read_block(source_block_id)
            .map(|result| {
                result.and_then(|block| {
                    (self.reduce_op)(block, &self.block_axes, self.transform.keepdims())
                })
            })
            .await?;

        futures::stream::iter(source_blocks.as_ref().iter().skip(1).copied())
            .map(|source_block_id| {
                self.source.read_block(source_block_id).map(|result| {
                    result.and_then(|block| {
                        (self.reduce_op)(block, &self.block_axes, self.transform.keepdims())
                    })
                })
            })
            .buffer_unordered(num_cpus::get())
            .try_fold(block, |block, source_block| async move {
                (self.reduce_blocks)(block, source_block)
            })
            .await
    }

    // TODO: optimize
    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);
        let blocks = futures::stream::iter(0..num_blocks)
            .map(move |block_id| {
                let this = self.clone();
                async move { this.read_block(block_id).await }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: TensorPermitRead, T: CDatatype> TensorPermitRead for DenseReduce<S, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<S: fmt::Debug, T: CDatatype> fmt::Debug for DenseReduce<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "reduce axes {:?} of {:?}",
            self.transform.reduce_axes(),
            self.source
        )
    }
}

#[derive(Clone)]
pub struct DenseReshape<S> {
    source: S,
    transform: Reshape,
}

impl<S: DenseInstance> DenseReshape<S> {
    pub fn new(source: S, shape: Shape) -> TCResult<Self> {
        Reshape::new(source.shape().clone(), shape).map(|transform| Self { source, transform })
    }
}

impl<S: TensorInstance> TensorInstance for DenseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseReshape<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Reshape:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Reshape;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let mut block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        let block = self.source.read_block(block_id).await?;

        if block.size() < self.block_size() {
            // this must be the trailing block
            let axis_dim = self.block_size() / block_shape.iter().skip(1).product::<usize>();
            block_shape[0] = axis_dim;
        }

        block.reshape(block_shape).map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let block_size = self.block_size();
        let block_axis = block_axis_for(self.shape(), block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), block_size);

        let source_blocks = self.source.read_blocks().await?;
        let blocks = source_blocks.map(move |result| {
            let block = result?;
            let mut block_shape = block_shape.to_vec();

            if block.size() < block_size {
                // this must be the trailing block
                let axis_dim = block_size / block_shape.iter().skip(1).product::<usize>();
                block_shape[0] = axis_dim;
            }

            block.reshape(block_shape).map_err(TCError::from)
        });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for DenseReshape<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        if range.is_empty() || range == Range::all(self.transform.shape()) {
            self.read_permit(txn_id, Range::default()).await
        } else {
            Err(bad_request!(
                "cannot lock range {:?} of {:?} for reading (consider making a copy first)",
                range,
                self
            ))
        }
    }
}

impl<FE, T: CDatatype, S: Into<DenseAccess<FE, T>>> From<DenseReshape<S>> for DenseAccess<FE, T> {
    fn from(reshape: DenseReshape<S>) -> Self {
        Self::Reshape(Box::new(DenseReshape {
            source: reshape.source.into(),
            transform: reshape.transform,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseReshape<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "reshape of {:?} into {:?}",
            self.source,
            self.transform.shape()
        )
    }
}

#[derive(Clone)]
pub struct DenseSlice<S> {
    source: S,
    transform: Slice,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseSlice<S> {
    pub fn new(source: S, range: Range) -> TCResult<Self> {
        let transform = Slice::new(source.shape().clone(), range)?;

        let block_axis = block_axis_for(source.shape(), source.block_size());
        let block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        let num_blocks = div_ceil(
            source.size() as u64,
            block_shape.iter().product::<usize>() as u64,
        ) as usize;

        let block_map_shape = source
            .shape()
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| {
                dim.try_into()
                    .map_err(|cause| bad_request!("invalid dimension: {dim} ({cause})"))
            })
            .collect::<Result<_, _>>()?;

        let block_map = ArrayBase::<Vec<_>>::new(
            block_map_shape,
            (0..num_blocks as u64).into_iter().collect(),
        )?;

        let mut block_map_bounds = Vec::with_capacity(block_axis + 1);
        for axis_range in transform.range().iter().take(block_axis).cloned() {
            let bound = axis_range.try_into()?;
            block_map_bounds.push(bound);
        }

        if transform.range().len() > block_axis {
            let bound = match &transform.range()[block_axis] {
                AxisRange::At(i) => {
                    let stride = block_map.shape().last().expect("stride");
                    let i = usize::try_from(*i)
                        .map_err(|cause| bad_request!("invalid index: {cause}"))?
                        / stride;
                    ha_ndarray::AxisBound::At(i)
                }
                AxisRange::In(axis_range, _step) => {
                    let stride = block_shape[0];
                    let start = usize::try_from(axis_range.start)
                        .map_err(|cause| bad_request!("invalid range start: {cause}"))?
                        / stride;
                    let stop = usize::try_from(axis_range.end)
                        .map_err(|cause| bad_request!("invalid range stop: {cause}"))?
                        / stride;
                    ha_ndarray::AxisBound::In(start, stop, 1)
                }
                AxisRange::Of(indices) => {
                    let stride = block_map.shape().last().expect("stride");
                    let indices = indices
                        .iter()
                        .copied()
                        .map(|i| {
                            usize::try_from(i)
                                .map(|i| i / stride)
                                .map_err(|cause| bad_request!("invalid index: {cause}"))
                        })
                        .collect::<Result<Vec<usize>, TCError>>()?;

                    ha_ndarray::AxisBound::Of(indices)
                }
            };

            block_map_bounds.push(bound);
        }

        let block_map = block_map.slice(block_map_bounds)?;
        let block_map = ArrayBase::<Vec<u64>>::copy(&block_map)?;

        let block_size = transform.shape().iter().product::<u64>() as usize / num_blocks;

        Ok(Self {
            source,
            transform,
            block_map,
            block_size,
        })
    }

    #[inline]
    fn block_bounds(&self, block_id: u64) -> TCResult<(u64, Vec<ha_ndarray::AxisBound>)> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;

        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let local_bound =
            match ha_ndarray::AxisBound::try_from(self.transform.range()[block_axis].clone())? {
                ha_ndarray::AxisBound::At(i) => ha_ndarray::AxisBound::At(i),
                ha_ndarray::AxisBound::In(start, stop, step) => {
                    let stride = block_shape[0];

                    if source_block_id == 0 {
                        ha_ndarray::AxisBound::In(start, stride, step)
                    } else if source_block_id == self.block_map.size() as u64 - 1 {
                        ha_ndarray::AxisBound::In(stop - (stop % stride), stop, step)
                    } else {
                        let start = source_block_id as usize * stride;
                        ha_ndarray::AxisBound::In(start, start + stride, step)
                    }
                }
                ha_ndarray::AxisBound::Of(indices) => {
                    if source_block_id < indices.len() as u64 {
                        let i = indices[source_block_id as usize] as usize;
                        ha_ndarray::AxisBound::At(i)
                    } else {
                        return Err(bad_request!("block id {} is out of range", block_id));
                    }
                }
            };

        let mut block_bounds = Vec::with_capacity(self.ndim());
        for bound in self.transform.range().iter().take(block_axis).cloned() {
            block_bounds.push(bound.try_into()?);
        }

        if block_bounds.is_empty() {
            block_bounds.push(local_bound);
        } else {
            block_bounds[0] = local_bound;
        }

        Ok((source_block_id, block_bounds))
    }
}

impl<S: DenseInstance + Clone> DenseSlice<S> {
    async fn block_stream<Get, Fut, Block>(
        self,
        get_block: Get,
    ) -> TCResult<impl Stream<Item = TCResult<Block::Slice>>>
    where
        Get: Fn(S, u64) -> Fut + Copy,
        Fut: Future<Output = TCResult<Block>>,
        Block: NDArrayTransform,
    {
        let ndim = self.ndim();
        let transform = self.transform;
        let range = transform.range();
        let block_map = self.block_map;
        let source = self.source;

        let block_axis = block_axis_for(transform.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, transform.shape(), self.block_size);

        let local_bounds = match ha_ndarray::AxisBound::try_from(range[block_axis].clone())? {
            ha_ndarray::AxisBound::At(i) => {
                debug_assert_eq!(block_map.size(), 1);
                vec![ha_ndarray::AxisBound::At(i)]
            }
            ha_ndarray::AxisBound::In(start, stop, step) => {
                let stride = block_shape[0];

                if block_map.size() == 1 {
                    vec![ha_ndarray::AxisBound::In(start, stop, step)]
                } else {
                    let mut local_bounds = Vec::with_capacity(block_map.size());
                    local_bounds.push(ha_ndarray::AxisBound::In(start, stride, step));

                    for i in 0..(block_map.size() - 2) {
                        let start = stride * i;
                        local_bounds.push(ha_ndarray::AxisBound::In(start, start + stride, step));
                    }

                    local_bounds.push(ha_ndarray::AxisBound::In(
                        stop - (stop % stride),
                        stop,
                        step,
                    ));

                    local_bounds
                }
            }
            ha_ndarray::AxisBound::Of(indices) => {
                indices.into_iter().map(ha_ndarray::AxisBound::At).collect()
            }
        };

        let mut block_bounds = Vec::<ha_ndarray::AxisBound>::with_capacity(ndim);
        for bound in range.iter().skip(block_axis).cloned() {
            block_bounds.push(bound.try_into()?);
        }

        debug_assert_eq!(block_map.size(), local_bounds.len());
        let blocks = stream::iter(block_map.into_inner().into_iter().zip(local_bounds))
            .map(move |(block_id, local_bound)| {
                let mut block_bounds = block_bounds.to_vec();
                let source = source.clone();

                async move {
                    let block = get_block(source, block_id).await?;

                    if block_bounds.is_empty() {
                        block_bounds.push(local_bound);
                    } else {
                        block_bounds[0] = local_bound;
                    }

                    block.slice(block_bounds).map_err(TCError::from)
                }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }
}

impl<S: TensorInstance> TensorInstance for DenseSlice<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseSlice<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Slice;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let (source_block_id, block_bounds) = self.block_bounds(block_id)?;
        let source_block = self.source.read_block(source_block_id).await?;
        source_block.slice(block_bounds).map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let blocks = self
            .block_stream(|source, block_id| async move { source.read_block(block_id).await })
            .await?;

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseSlice<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

#[async_trait]
impl<S: TensorPermitWrite> TensorPermitWrite for DenseSlice<S> {
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.write_permit(txn_id, range).await
    }
}

#[async_trait]
impl<'a, S: DenseWrite + Clone> DenseWrite for DenseSlice<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
    S::BlockWrite: NDArrayTransform,
    <S::BlockWrite as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + NDArrayWrite + Into<Array<S::DType>>,
{
    type BlockWrite = <S::BlockWrite as NDArrayTransform>::Slice;

    async fn write_block(&self, block_id: u64) -> TCResult<Self::BlockWrite> {
        let (source_block_id, block_bounds) = self.block_bounds(block_id)?;
        let source_block = self.source.write_block(source_block_id).await?;
        source_block.slice(block_bounds).map_err(TCError::from)
    }

    async fn write_blocks(self) -> TCResult<BlockStream<Self::BlockWrite>> {
        let blocks = self
            .block_stream(|source, block_id| async move { source.write_block(block_id).await })
            .await?;

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<'a, S: DenseWrite + DenseWriteLock<'a> + Clone> DenseWriteLock<'a> for DenseSlice<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
    S::BlockWrite: NDArrayTransform,
    <S::BlockWrite as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + NDArrayWrite + Into<Array<S::DType>>,
{
    type WriteGuard = DenseSliceWriteGuard<'a, S>;

    async fn write(&'a self) -> Self::WriteGuard {
        DenseSliceWriteGuard { dest: self }
    }
}

impl<FE, T: CDatatype, S: Into<DenseAccess<FE, T>>> From<DenseSlice<S>> for DenseAccess<FE, T> {
    fn from(slice: DenseSlice<S>) -> Self {
        Self::Slice(Box::new(DenseSlice {
            source: slice.source.into(),
            transform: slice.transform,
            block_map: slice.block_map,
            block_size: slice.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "slice {:?} from {:?}",
            self.transform.range(),
            self.source
        )
    }
}

pub struct DenseSliceWriteGuard<'a, S> {
    dest: &'a DenseSlice<S>,
}

#[async_trait]
impl<'a, S> DenseWriteGuard<S::DType> for DenseSliceWriteGuard<'a, S>
where
    S: DenseWrite + DenseWriteLock<'a> + Clone,
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
    S::BlockWrite: NDArrayTransform,
    <S::BlockWrite as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + NDArrayWrite + Into<Array<S::DType>>,
{
    async fn overwrite<O: DenseInstance<DType = S::DType>>(&self, other: O) -> TCResult<()> {
        let block_axis = block_axis_for(self.dest.shape(), self.dest.block_size);
        let block_shape = block_shape_for(block_axis, self.dest.shape(), self.dest.block_size);

        let dest = self.dest.clone().write_blocks().await?;
        let source = other.read_blocks().await?;
        let source = BlockResize::new(source, block_shape)?;

        dest.zip(source)
            .map(|(dest, source)| {
                let mut dest = dest?;
                let source = source?;
                dest.write(&source).map_err(TCError::from)
            })
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, value: S::DType) -> TCResult<()> {
        let dest = self.dest.clone().write_blocks().await?;
        dest.map_ok(|mut block| block.write_value(value))
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, coord: Coord, value: S::DType) -> TCResult<()> {
        let source_coord = self.dest.transform.invert_coord(coord)?;
        let source = self.dest.source.write().await;
        source.write_value(source_coord, value).await
    }
}

#[derive(Clone)]
pub struct DenseTranspose<S> {
    source: S,
    transform: Transpose,
    block_map: ArrayBase<Vec<u64>>,
    block_axes: Axes,
}

impl<S: DenseInstance> DenseTranspose<S> {
    pub fn new(source: S, permutation: Option<Axes>) -> TCResult<Self> {
        let transform = Transpose::new(source.shape().clone(), permutation)?;

        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());

        let map_shape = source
            .shape()
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect();

        let permutation = transform.axes().to_vec();
        let (map_axes, block_axes) = permutation.split_at(block_axis);

        if map_axes.iter().copied().any(|x| x >= block_axis)
            || block_axes.iter().copied().any(|x| x <= block_axis)
        {
            return Err(bad_request!(
                "cannot transpose axes {:?} of {:?} without copying",
                block_axes,
                source
            ));
        }

        let block_map = ArrayBase::<Vec<_>>::new(map_shape, (0..num_blocks).into_iter().collect())?;
        let block_map = block_map.transpose(Some(map_axes.to_vec()))?;
        let block_map = ArrayBase::<Vec<_>>::copy(&block_map)?;

        Ok(Self {
            source,
            transform,
            block_map,
            block_axes: block_axes.to_vec(),
        })
    }
}

impl<S: TensorInstance> TensorInstance for DenseTranspose<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseTranspose<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Transpose:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Transpose;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block = self.source.read_block(source_block_id).await?;

        block
            .transpose(Some(self.transform.axes().to_vec()))
            .map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let block_axes = self.block_axes;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(block_id).await }
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let block = result?;

                block
                    .transpose(Some(block_axes.to_vec()))
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseTranspose<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(&range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<FE, T: CDatatype, S: Into<DenseAccess<FE, T>>> From<DenseTranspose<S>> for DenseAccess<FE, T> {
    fn from(transpose: DenseTranspose<S>) -> Self {
        Self::Transpose(Box::new(DenseTranspose {
            source: transpose.source.into(),
            transform: transpose.transform,
            block_map: transpose.block_map,
            block_axes: transpose.block_axes,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseTranspose<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "transpose axes {:?} of {:?}",
            self.transform.axes(),
            self.source
        )
    }
}

#[derive(Clone)]
pub struct DenseUnary<S, T: CDatatype> {
    source: S,
    op: fn(Array<T>) -> TCResult<Array<T>>,
}

impl<S: DenseInstance> DenseUnary<S, S::DType> {
    fn new(source: S, op: fn(Array<S::DType>) -> TCResult<Array<S::DType>>) -> Self {
        Self { source, op }
    }

    pub fn abs(source: S) -> Self {
        Self::new(source, |block| {
            block.abs().map(Array::from).map_err(TCError::from)
        })
    }

    pub fn exp(source: S) -> Self {
        Self::new(source, |block| {
            block.exp().map(Array::from).map_err(TCError::from)
        })
    }

    pub fn ln(source: S) -> Self {
        Self::new(source, |block| {
            block.ln().map(Array::from).map_err(TCError::from)
        })
    }

    pub fn round(source: S) -> Self {
        Self::new(source, |block| {
            block.round().map(Array::from).map_err(TCError::from)
        })
    }
}

impl<S: TensorInstance, T: CDatatype> TensorInstance for DenseUnary<S, T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseUnary<S, S::DType>
where
    Array<S::DType>: From<S::Block>,
{
    type Block = Array<S::DType>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.source
            .read_block(block_id)
            .map_ok(Array::from)
            .map(move |result| result.and_then(|block| (self.op)(block)))
            .await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let source_blocks = self.source.read_blocks().await?;
        let blocks = source_blocks
            .map_ok(Array::from)
            .map(move |result| result.and_then(|block| (self.op)(block)));

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: TensorPermitRead, T: CDatatype> TensorPermitRead for DenseUnary<S, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<S: fmt::Debug, T: CDatatype> fmt::Debug for DenseUnary<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary transform of {:?}", self.source)
    }
}

pub struct DenseUnaryBoolean<FE, T: CDatatype> {
    source: DenseCastSource<FE>,
    op: fn(ArrayCastSource) -> TCResult<Array<T>>,
}

impl<FE, T: CDatatype> Clone for DenseUnaryBoolean<FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            op: self.op,
        }
    }
}

impl<FE> DenseUnaryBoolean<FE, u8> {
    pub fn not<S: Into<DenseCastSource<FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            op: ArrayCastSource::not,
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for DenseUnaryBoolean<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let source = &self.source;
        cast_dispatch!(source, this, this.shape())
    }
}

#[async_trait]
impl<FE: DenseCacheFile, T: CDatatype + DType> DenseInstance for DenseUnaryBoolean<FE, T> {
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let source = &self.source;
        cast_dispatch!(source, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.source
            .read_block(block_id)
            .map(move |result| result.and_then(|block| (self.op)(block)))
            .await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let source_blocks = self.source.read_blocks().await?;
        let blocks = source_blocks.map(move |result| result.and_then(|block| (self.op)(block)));
        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<FE: Send + Sync + 'static, T: CDatatype> TensorPermitRead for DenseUnaryBoolean<FE, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        let source = &self.source;
        cast_dispatch!(source, this, this.read_permit(txn_id, range).await)
    }
}

impl<FE, T: CDatatype> fmt::Debug for DenseUnaryBoolean<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary boolean transform of {:?}", self.source)
    }
}

#[inline]
fn block_axis_for(shape: &[u64], block_size: usize) -> usize {
    debug_assert!(!shape.is_empty());
    debug_assert!(shape.iter().copied().all(|dim| dim > 0));
    debug_assert!(shape.iter().product::<u64>() >= block_size as u64);

    let mut block_ndim = 1;
    let mut size = 1;
    for dim in shape.iter().rev() {
        size *= dim;

        if size > block_size as u64 {
            break;
        } else {
            block_ndim += 1;
        }
    }

    shape.len() - block_ndim
}

#[inline]
fn block_shape_for(axis: usize, shape: &[u64], block_size: usize) -> BlockShape {
    if axis == shape.len() - 1 {
        vec![block_size]
    } else {
        let axis_dim = (shape.iter().skip(axis).product::<u64>() / block_size as u64) as usize;
        debug_assert_eq!(block_size % axis_dim, 0);

        let mut block_shape = BlockShape::with_capacity(shape.len() - axis + 1);
        block_shape.push(axis_dim);
        block_shape.extend(shape.iter().skip(axis).copied().map(|dim| dim as usize));

        debug_assert!(!block_shape.is_empty());

        block_shape
    }
}

#[inline]
fn div_ceil(num: u64, denom: u64) -> u64 {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
    }
}

#[inline]
fn source_block_id_for(block_map: &ArrayBase<Vec<u64>>, block_id: u64) -> TCResult<u64> {
    block_map
        .as_slice()
        .get(block_id as usize)
        .copied()
        .ok_or_else(|| bad_request!("block id {} is out of range", block_id))
}
