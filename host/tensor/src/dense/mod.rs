use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;

use afarray::{Array, ArrayInstance};
use arrayfire as af;
use async_trait::async_trait;
use destream::{de, en, EncodeSeq};
use futures::future::TryFutureExt;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use number_general::{Number, NumberType};

use tc_error::*;
use tc_transact::fs::{BlockData, Dir, File};
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::ValueType;
use tcgeneric::{NativeClass, TCBoxTryFuture, TCPathBuf, TCTryStream};

use super::{Bounds, Coord, Read, ReadValueAt, Shape, TensorAccess, TensorIO, TensorType};

pub use file::BlockListFile;

mod file;

// = 1 mibibyte / 64 bits (must be the same as Array::max_size())
const PER_BLOCK: usize = 131_072;

#[async_trait]
pub trait DenseAccess<F: File<Array>, D: Dir, T: Transaction<D>>:
    ReadValueAt<D, Txn = T> + TensorAccess + Send + Sync + Sized + 'static
{
    type Slice: Clone + DenseAccess<F, D, T>;

    fn accessor(self) -> DenseAccessor<F, D, T>;

    fn block_stream<'a>(self, txn: Self::Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let blocks = self
                .value_stream(txn)
                .await?
                .chunks(Array::max_size() as usize)
                .map(|values| values.into_iter().collect::<TCResult<Vec<Number>>>())
                .map_ok(Array::from);

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn value_stream<'a>(self, txn: Self::Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = self.block_stream(txn).await?;

            let values = values
                .map_ok(|array| array.to_vec())
                .map_ok(|values| {
                    values
                        .into_iter()
                        .map(Ok)
                        .collect::<Vec<TCResult<Number>>>()
                })
                .map_ok(futures::stream::iter)
                .try_flatten();

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice>;

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()>;

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()>;
}

#[derive(Clone)]
pub enum DenseAccessor<F, D, T> {
    File(BlockListFile<F, D, T>),
    Slice(file::BlockListFileSlice<F, D, T>),
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> TensorAccess for DenseAccessor<F, D, T> {
    fn dtype(&self) -> NumberType {
        match self {
            Self::File(file) => file.dtype(),
            Self::Slice(slice) => slice.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::File(file) => file.ndim(),
            Self::Slice(slice) => slice.ndim(),
        }
    }

    fn shape(&self) -> &Shape {
        match self {
            Self::File(file) => file.shape(),
            Self::Slice(slice) => slice.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::File(file) => file.size(),
            Self::Slice(slice) => slice.size(),
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> DenseAccess<F, D, T> for DenseAccessor<F, D, T> {
    type Slice = Self;

    fn accessor(self) -> Self {
        self
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        match self {
            Self::File(file) => file.block_stream(txn),
            Self::Slice(slice) => slice.block_stream(txn),
        }
    }

    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        match self {
            Self::File(file) => file.value_stream(txn),
            Self::Slice(slice) => slice.value_stream(txn),
        }
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self> {
        match self {
            Self::File(file) => file.slice(bounds).map(Self::Slice),
            Self::Slice(slice) => slice.slice(bounds).map(Self::Slice),
        }
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        match self {
            Self::File(file) => file.write_value(txn_id, bounds, number).await,
            Self::Slice(slice) => slice.write_value(txn_id, bounds, number).await,
        }
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        match self {
            Self::File(file) => file.write_value_at(txn_id, coord, value),
            Self::Slice(slice) => slice.write_value_at(txn_id, coord, value),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> ReadValueAt<D> for DenseAccessor<F, D, T> {
    type Txn = T;

    fn read_value_at<'a>(&'a self, txn: &'a T, coord: Coord) -> Read<'a> {
        match self {
            Self::File(file) => file.read_value_at(txn, coord),
            Self::Slice(slice) => slice.read_value_at(txn, coord),
        }
    }
}

impl<F, D, T> From<BlockListFile<F, D, T>> for DenseAccessor<F, D, T> {
    fn from(file: BlockListFile<F, D, T>) -> Self {
        Self::File(file)
    }
}

#[derive(Clone)]
pub struct DenseTensor<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> {
    blocks: B,
    file: PhantomData<F>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseTensor<F, D, T, B> {
    pub fn into_inner(self) -> B {
        self.blocks
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> DenseTensor<F, D, T, BlockListFile<F, D, T>> {
    pub async fn constant<S>(file: F, txn_id: TxnId, shape: S, value: Number) -> TCResult<Self>
    where
        Shape: From<S>,
    {
        BlockListFile::constant(file, txn_id, shape.into(), value)
            .map_ok(Self::from)
            .await
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for DenseTensor<F, D, T, B>
{
    fn dtype(&self) -> NumberType {
        self.blocks.dtype()
    }

    fn ndim(&self) -> usize {
        self.blocks.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.blocks.shape()
    }

    fn size(&self) -> u64 {
        self.blocks.size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: Clone + DenseAccess<F, D, T>> TensorIO<D>
    for DenseTensor<F, D, T, B>
{
    type Txn = T;

    async fn read_value(&self, txn: &Self::Txn, coord: Coord) -> TCResult<Number> {
        self.blocks
            .read_value_at(txn, coord.to_vec())
            .map_ok(|(_, val)| val)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        self.blocks.clone().write_value(txn_id, bounds, value).await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.blocks.write_value_at(txn_id, coord, value).await
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: Clone + DenseAccess<F, D, T>> ReadValueAt<D>
    for DenseTensor<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(&'a self, txn: &'a T, coord: Coord) -> Read<'a> {
        self.blocks.read_value_at(txn, coord)
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> From<B>
    for DenseTensor<F, D, T, B>
{
    fn from(blocks: B) -> Self {
        Self {
            blocks,
            file: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> de::FromStream
    for DenseTensor<F, D, T, BlockListFile<F, D, T>>
where
    <D as Dir>::FileClass: From<TensorType> + Send,
    F: TryFrom<<D as Dir>::File, Error = TCError>,
{
    type Context = T;

    async fn from_stream<De: de::Decoder>(txn: T, decoder: &mut De) -> Result<Self, De::Error> {
        let txn_id = *txn.id();
        let file = txn
            .context()
            .create_file_tmp(txn_id, TensorType::Dense)
            .map_err(de::Error::custom)
            .await?;

        decoder
            .decode_seq(DenseTensorVisitor::new(txn_id, file))
            .await
    }
}

struct DenseTensorVisitor<F, D, T> {
    txn_id: TxnId,
    file: F,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F, D, T> DenseTensorVisitor<F, D, T> {
    fn new(txn_id: TxnId, file: F) -> Self {
        Self {
            txn_id,
            file,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> de::Visitor for DenseTensorVisitor<F, D, T> {
    type Value = DenseTensor<F, D, T, BlockListFile<F, D, T>>;

    fn expecting() -> &'static str {
        "a dense tensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let (dtype, shape) = seq
            .next_element::<(ValueType, Vec<u64>)>(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a tensor schema"))?;

        let dtype = dtype
            .try_into()
            .map_err(|_| de::Error::invalid_type(dtype, "a Number type"))?;

        let cxt = (self.txn_id, self.file, (dtype, shape.into()));
        let blocks = seq
            .next_element::<BlockListFile<F, D, T>>(cxt)
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "dense tensor data"))?;

        Ok(DenseTensor::from(blocks))
    }
}

#[async_trait]
impl<'en, F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> IntoView<'en, D>
    for DenseTensor<F, D, T, B>
{
    type Txn = T;
    type View = DenseTensorView<'en>;

    async fn into_view(self, txn: T) -> TCResult<DenseTensorView<'en>> {
        let dtype = self.dtype();
        let shape = self.shape().to_vec();
        let blocks = self.blocks.block_stream(txn).await?;

        Ok(DenseTensorView {
            schema: (ValueType::from(dtype).path(), shape),
            blocks: BlockStreamView { dtype, blocks },
        })
    }
}

pub struct DenseTensorView<'en> {
    schema: (TCPathBuf, Vec<u64>),
    blocks: BlockStreamView<'en>,
}

#[async_trait]
impl<'en> en::IntoStream<'en> for DenseTensorView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let mut seq = encoder.encode_seq(Some(2))?;
        seq.encode_element(self.schema)?;
        seq.encode_element(self.blocks)?;
        seq.end()
    }
}

struct BlockStreamView<'en> {
    dtype: NumberType,
    blocks: TCTryStream<'en, Array>,
}

impl<'en> en::IntoStream<'en> for BlockStreamView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use number_general::{
            ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT,
        };

        fn encodable<'en, E: en::Error + 'en, T: af::HasAfEnum + Clone + Default + 'en>(
            blocks: TCTryStream<'en, Array>,
        ) -> impl Stream<Item = Result<Vec<T>, E>> + 'en {
            blocks
                .map_ok(|arr| arr.type_cast())
                .map_ok(|arr| arr.to_vec())
                .map_err(en::Error::custom)
        }

        match self.dtype {
            NT::Bool => encoder.encode_array_bool(encodable(self.blocks)),
            NT::Complex(ct) => match ct {
                CT::C32 => encoder.encode_array_f32(encodable_c32(self.blocks)),
                _ => encoder.encode_array_f64(encodable_c64(self.blocks)),
            },
            NT::Float(ft) => match ft {
                FT::F32 => encoder.encode_array_f32(encodable(self.blocks)),
                _ => encoder.encode_array_f64(encodable(self.blocks)),
            },
            NT::Int(it) => match it {
                IT::I8 | IT::I16 => encoder.encode_array_i16(encodable(self.blocks)),
                IT::I32 => encoder.encode_array_i32(encodable(self.blocks)),
                _ => encoder.encode_array_i64(encodable(self.blocks)),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => encoder.encode_array_u8(encodable(self.blocks)),
                UT::U16 => encoder.encode_array_u16(encodable(self.blocks)),
                UT::U32 => encoder.encode_array_u32(encodable(self.blocks)),
                _ => encoder.encode_array_u64(encodable(self.blocks)),
            },
            NT::Number => Err(en::Error::custom(format!(
                "invalid Tensor data type: {}",
                NT::Number
            ))),
        }
    }
}

fn encodable_c32<'en, E: en::Error + 'en>(
    blocks: TCTryStream<'en, Array>,
) -> impl Stream<Item = Result<Vec<f32>, E>> + 'en {
    blocks
        .map_ok(|arr| {
            let source = arr.type_cast::<afarray::Complex<f32>>();
            let re = source.re();
            let im = source.im();

            let mut i = 0;
            let mut dest = vec![0.; source.len() * 2];
            for (re, im) in re.to_vec().into_iter().zip(im.to_vec()) {
                dest[i] = re;
                dest[i + 1] = im;
                i += 2;
            }

            dest
        })
        .map_err(en::Error::custom)
}

fn encodable_c64<'en, E: en::Error + 'en>(
    blocks: TCTryStream<'en, Array>,
) -> impl Stream<Item = Result<Vec<f64>, E>> + 'en {
    blocks
        .map_ok(|arr| {
            let source = arr.type_cast::<afarray::Complex<f64>>();
            let re = source.re();
            let im = source.im();

            let mut i = 0;
            let mut dest = vec![0.; source.len() * 2];
            for (re, im) in re.to_vec().into_iter().zip(im.to_vec()) {
                dest[i] = re;
                dest[i + 1] = im;
                i += 2;
            }

            dest
        })
        .map_err(en::Error::custom)
}

fn block_offsets(
    af_indices: &af::Array<u64>,
    af_offsets: &af::Array<u64>,
    start: f64,
    block_id: u64,
) -> (af::Array<u64>, f64) {
    assert_eq!(af_indices.elements(), af_offsets.elements());

    let num_to_update = af::sum_all(&af::eq(
        af_indices,
        &af::constant(block_id, af::Dim4::new(&[1, 1, 1, 1])),
        true,
    ))
    .0;

    if num_to_update == 0 {
        return (af::Array::new_empty(af::Dim4::default()), start);
    }

    debug_assert!((start as usize + num_to_update as usize) <= af_offsets.elements());

    let num_to_update = num_to_update as f64;
    let block_offsets = af::index(
        af_offsets,
        &[af::Seq::new(start, (start + num_to_update) - 1f64, 1f64)],
    );

    (block_offsets, (start + num_to_update))
}

fn coord_block<I: Iterator<Item = Coord>>(
    coords: I,
    coord_bounds: &[u64],
    per_block: usize,
    ndim: usize,
    num_coords: u64,
) -> (Coord, af::Array<u64>, af::Array<u64>) {
    let coords: Vec<u64> = coords.flatten().collect();
    assert!(coords.len() > 0);
    assert!(ndim > 0);

    let af_per_block = af::constant(per_block as u64, af::Dim4::new(&[1, 1, 1, 1]));
    let af_coord_bounds = af::Array::new(coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));

    let af_coords = af::Array::new(
        &coords,
        af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]),
    );
    let af_coords = af::mul(&af_coords, &af_coord_bounds, true);
    let af_coords = af::sum(&af_coords, 0);

    let af_offsets = af::modulo(&af_coords, &af_per_block, true);
    let af_indices = af_coords / af_per_block;

    let af_block_ids = af::set_unique(&af_indices, true);
    let mut block_ids = vec![0u64; af_block_ids.elements()];
    af_block_ids.host(&mut block_ids);
    (block_ids, af_indices, af_offsets)
}
