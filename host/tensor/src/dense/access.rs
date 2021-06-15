use afarray::Array;
use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::try_join;
use number_general::*;

use tc_error::*;
use tc_transact::fs::{Dir, File};
use tc_transact::{Transaction, TxnId};
use tcgeneric::{TCBoxTryFuture, TCStream, TCTryStream};

use crate::stream::{Read, ReadValueAt};
use crate::transform::{self, Rebase};
use crate::{Bounds, Coord, Phantom, Shape, TensorAccess};

use super::{DenseAccess, DenseAccessor, DenseTensor};

const ERR_NONBIJECTIVE_WRITE: &str = "cannot write to a derived tensor which is \
not a bijection of its source";

#[derive(Clone)]
pub struct BlockListCombine<F, D, T, L, R> {
    left: L,
    right: R,
    combinator: fn(&Array, &Array) -> Array,
    value_combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
    phantom: Phantom<F, D, T>,
}

impl<
        F: File<Array>,
        D: Dir,
        T: Transaction<D>,
        L: DenseAccess<F, D, T>,
        R: DenseAccess<F, D, T>,
    > BlockListCombine<F, D, T, L, R>
{
    pub fn new(
        left: L,
        right: R,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<Self> {
        if left.shape() != right.shape() {
            return Err(TCError::bad_request(
                format!("Cannot combine shape {} with shape", left.shape()),
                right.shape(),
            ));
        }

        Ok(BlockListCombine {
            left,
            right,
            combinator,
            value_combinator,
            dtype,
            phantom: Phantom::default(),
        })
    }
}

impl<
        F: File<Array>,
        D: Dir,
        T: Transaction<D>,
        L: DenseAccess<F, D, T>,
        R: DenseAccess<F, D, T>,
    > TensorAccess for BlockListCombine<F, D, T, L, R>
{
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.left.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.left.shape()
    }

    fn size(&self) -> u64 {
        self.left.size()
    }
}

#[async_trait]
impl<
        F: File<Array>,
        D: Dir,
        T: Transaction<D>,
        L: DenseAccess<F, D, T>,
        R: DenseAccess<F, D, T>,
    > DenseAccess<F, D, T> for BlockListCombine<F, D, T, L, R>
where
    Self: Clone,
{
    type Slice = BlockListCombine<
        F,
        D,
        T,
        <L as DenseAccess<F, D, T>>::Slice,
        <R as DenseAccess<F, D, T>>::Slice,
    >;
    type Transpose = BlockListCombine<
        F,
        D,
        T,
        <L as DenseAccess<F, D, T>>::Transpose,
        <R as DenseAccess<F, D, T>>::Transpose,
    >;

    fn accessor(self) -> DenseAccessor<F, D, T> {
        let left = self.left.accessor();
        let right = self.right.accessor();
        let combine = BlockListCombine {
            left,
            right,
            combinator: self.combinator,
            value_combinator: self.value_combinator,
            dtype: self.dtype,
            phantom: Phantom::default(),
        };

        DenseAccessor::Combine(Box::new(combine))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let left = self.left.block_stream(txn.clone());
            let right = self.right.block_stream(txn);
            let (left, right) = try_join!(left, right)?;

            let combinator = self.combinator;
            let blocks = left
                .zip(right)
                .map(|(l, r)| Ok((l?, r?)))
                .map_ok(move |(l, r)| combinator(&l, &r));

            let blocks: TCTryStream<'a, Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let left = self.left.slice(bounds.clone())?;
        let right = self.right.slice(bounds)?;

        BlockListCombine::new(
            left,
            right,
            self.combinator,
            self.value_combinator,
            self.dtype,
        )
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let left = self.left.transpose(permutation.clone())?;
        let right = self.right.transpose(permutation)?;

        BlockListCombine::new(
            left,
            right,
            self.combinator,
            self.value_combinator,
            self.dtype,
        )
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(TCError::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<
        F: File<Array>,
        D: Dir,
        T: Transaction<D>,
        L: DenseAccess<F, D, T>,
        R: DenseAccess<F, D, T>,
    > ReadValueAt<D> for BlockListCombine<F, D, T, L, R>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let left = self.left.read_value_at(txn.clone(), coord.to_vec());
            let right = self.right.read_value_at(txn, coord);
            let ((coord, left), (_, right)) = try_join!(left, right)?;
            let value = (self.value_combinator)(left, right);
            Ok((coord, value))
        })
    }
}

#[derive(Clone)]
pub struct BlockListBroadcast<F, D, T, B> {
    source: B,
    rebase: transform::Broadcast,
    phantom: Phantom<F, D, T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    BlockListBroadcast<F, D, T, B>
{
    pub fn new(source: B, shape: Shape) -> TCResult<Self> {
        let rebase = transform::Broadcast::new(source.shape().clone(), shape)?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for BlockListBroadcast<F, D, T, B>
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseAccess<F, D, T>
    for BlockListBroadcast<F, D, T, B>
{
    type Slice = BlockListBroadcast<F, D, T, <B as DenseAccess<F, D, T>>::Slice>;
    type Transpose = BlockListTranspose<F, D, T, Self>;

    fn accessor(self) -> DenseAccessor<F, D, T> {
        let source = self.source.accessor();
        let broadcast = BlockListBroadcast {
            source,
            rebase: self.rebase,
            phantom: Phantom::default(),
        };

        DenseAccessor::Broadcast(Box::new(broadcast))
    }

    // TODO: replace with block_stream
    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        let values = stream::iter(Bounds::all(self.shape()).affected()).then(move |coord| {
            self.clone()
                .read_value_at(txn.clone(), coord)
                .map_ok(|(_, value)| value)
        });

        let values: TCTryStream<'a, Number> = Box::pin(values);
        Box::pin(future::ready(Ok(values)))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        self.shape().validate_bounds(&bounds)?;

        let shape = bounds.to_shape();
        let bounds = self.rebase.invert_bounds(bounds);
        let source = self.source.slice(bounds)?;
        BlockListBroadcast::new(source, shape)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        BlockListTranspose::new(self, permutation)
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(TCError::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> ReadValueAt<D>
    for BlockListBroadcast<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct BlockListCast<F, D, T, B> {
    source: B,
    dtype: NumberType,
    phantom: Phantom<F, D, T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> BlockListCast<F, D, T, B> {
    pub fn new(source: B, dtype: NumberType) -> Self {
        Self {
            source,
            dtype,
            phantom: Phantom::default(),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for BlockListCast<F, D, T, B>
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseAccess<F, D, T>
    for BlockListCast<F, D, T, B>
{
    type Slice = BlockListCast<F, D, T, <B as DenseAccess<F, D, T>>::Slice>;
    type Transpose = BlockListCast<F, D, T, <B as DenseAccess<F, D, T>>::Transpose>;

    fn accessor(self) -> DenseAccessor<F, D, T> {
        let cast = BlockListCast::new(self.source.accessor(), self.dtype);
        DenseAccessor::Cast(Box::new(cast))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let dtype = self.dtype;
            let blocks: TCStream<'a, TCResult<Array>> = self.source.block_stream(txn).await?;
            let cast = blocks.map_ok(move |array| array.cast_into(dtype));
            let cast: TCTryStream<'a, Array> = Box::pin(cast);
            Ok(cast)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let slice = self.source.slice(bounds)?;
        Ok(BlockListCast::new(slice, self.dtype))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let transpose = self.source.transpose(permutation)?;
        Ok(BlockListCast::new(transpose, self.dtype))
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        self.source.write_value_at(txn_id, coord, value)
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> ReadValueAt<D>
    for BlockListCast<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct BlockListExpand<F, D, T, B> {
    source: B,
    rebase: transform::Expand,
    phantom: Phantom<F, D, T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    BlockListExpand<F, D, T, B>
{
    pub fn new(source: B, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(source.shape().clone(), axis)?;
        Ok(Self {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for BlockListExpand<F, D, T, B>
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim() + 1
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.shape().size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseAccess<F, D, T>
    for BlockListExpand<F, D, T, B>
{
    type Slice = <B as DenseAccess<F, D, T>>::Slice;
    type Transpose = <B as DenseAccess<F, D, T>>::Transpose;

    fn accessor(self) -> DenseAccessor<F, D, T> {
        let expand = BlockListExpand {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        };

        DenseAccessor::Expand(Box::new(expand))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        self.source.block_stream(txn)
    }

    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        self.source.value_stream(txn)
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.slice(bounds) // TODO: expand the result
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let permutation = permutation.map(|axes| self.rebase.invert_axes(axes));
        self.source.transpose(permutation) // TODO: expand the result
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> ReadValueAt<D>
    for BlockListExpand<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, value)| (coord, value));

        Box::pin(read)
    }
}

// TODO: &Txn, not Txn
type Reductor<F, D, T> =
    fn(&DenseTensor<F, D, T, DenseAccessor<F, D, T>>, T) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct BlockListReduce<F, D, T, B> {
    source: B,
    rebase: transform::Reduce,
    reductor: Reductor<F, D, T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    BlockListReduce<F, D, T, B>
{
    pub fn new(source: B, axis: usize, reductor: Reductor<F, D, T>) -> TCResult<Self> {
        let rebase = transform::Reduce::new(source.shape().clone(), axis)?;

        Ok(BlockListReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for BlockListReduce<F, D, T, B>
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.shape().size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseAccess<F, D, T>
    for BlockListReduce<F, D, T, B>
{
    type Slice = BlockListReduce<F, D, T, <B as DenseAccess<F, D, T>>::Slice>;
    type Transpose = BlockListReduce<F, D, T, <B as DenseAccess<F, D, T>>::Transpose>;

    fn accessor(self) -> DenseAccessor<F, D, T> {
        let reduce = BlockListReduce {
            source: self.source.accessor(),
            rebase: self.rebase,
            reductor: self.reductor,
        };

        DenseAccessor::Reduce(Box::new(reduce))
    }

    fn value_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = stream::iter(Bounds::all(self.shape()).affected()).then(move |coord| {
                let txn = txn.clone();
                let source = self.source.clone();
                let reductor = self.reductor;
                let source_bounds = self.rebase.invert_coord(&coord);
                Box::pin(async move {
                    let slice = source.slice(source_bounds)?;
                    reductor(&slice.accessor().into(), txn.clone()).await
                })
            });

            let values: TCTryStream<'a, Number> = Box::pin(values);
            Ok(values)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let reduce_axis = self.rebase.reduce_axis(&bounds);
        let source_bounds = self.rebase.invert_bounds(bounds);
        let slice = self.source.slice(source_bounds)?;
        BlockListReduce::new(slice, reduce_axis, self.reductor)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let (source_axes, reduce_axis) = self.rebase.invert_axes(permutation);
        let source = self.source.transpose(Some(source_axes))?;
        BlockListReduce::new(source, reduce_axis, self.reductor)
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(TCError::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> ReadValueAt<D>
    for BlockListReduce<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let reductor = self.reductor;
            let source_bounds = self.rebase.invert_coord(&coord);
            let slice = self.source.slice(source_bounds)?;
            let value = reductor(&slice.accessor().into(), txn.clone()).await?;

            Ok((coord, value))
        })
    }
}

#[derive(Clone)]
pub struct BlockListTranspose<F, D, T, B> {
    source: B,
    rebase: transform::Transpose,
    phantom: Phantom<F, D, T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    BlockListTranspose<F, D, T, B>
{
    pub fn new(source: B, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        let rebase = transform::Transpose::new(source.shape().clone(), permutation)?;
        Ok(BlockListTranspose {
            source,
            rebase,
            phantom: Phantom::default(),
        })
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for BlockListTranspose<F, D, T, B>
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseAccess<F, D, T>
    for BlockListTranspose<F, D, T, B>
{
    type Slice = BlockListTranspose<F, D, T, <B as DenseAccess<F, D, T>>::Slice>;
    type Transpose = Self;

    fn accessor(self) -> DenseAccessor<F, D, T> {
        let accessor = BlockListTranspose {
            source: self.source.accessor(),
            rebase: self.rebase,
            phantom: Phantom::default(),
        };

        DenseAccessor::Transpose(Box::new(accessor))
    }

    fn block_stream<'a>(self, _txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(future::ready(Err(TCError::not_implemented(
            "BlockListTranspose::block_stream",
        ))))
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice> {
        Err(TCError::not_implemented("BlockListTranspose::slice"))
    }

    fn transpose(self, _permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        Err(TCError::not_implemented("BlockListTranspose::transpose"))
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> ReadValueAt<D>
    for BlockListTranspose<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));

        Box::pin(read)
    }
}

#[derive(Clone)]
pub struct BlockListUnary<F, D, T, B> {
    source: B,
    transform: fn(&Array) -> Array,
    value_transform: fn(Number) -> Number,
    dtype: NumberType,
    phantom: Phantom<F, D, T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>>
    BlockListUnary<F, D, T, B>
{
    pub fn new(
        source: B,
        transform: fn(&Array) -> Array,
        value_transform: fn(Number) -> Number,
        dtype: NumberType,
    ) -> Self {
        Self {
            source,
            transform,
            value_transform,
            dtype,
            phantom: Phantom::default(),
        }
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> TensorAccess
    for BlockListUnary<F, D, T, B>
{
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.source.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.source.shape()
    }

    fn size(&self) -> u64 {
        self.source.size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> DenseAccess<F, D, T>
    for BlockListUnary<F, D, T, B>
{
    type Slice = BlockListUnary<F, D, T, <B as DenseAccess<F, D, T>>::Slice>;
    type Transpose = BlockListUnary<F, D, T, <B as DenseAccess<F, D, T>>::Transpose>;

    fn accessor(self) -> DenseAccessor<F, D, T> {
        let unary = BlockListUnary::new(
            self.source.accessor(),
            self.transform,
            self.value_transform,
            self.dtype,
        );

        DenseAccessor::Unary(Box::new(unary))
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let transform = self.transform;
            let blocks = self.source.block_stream(txn).await?;
            let blocks: TCTryStream<'a, Array> =
                Box::pin(blocks.map_ok(move |array| transform(&array)));

            Ok(blocks)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let source = self.source.slice(bounds)?;
        Ok(BlockListUnary {
            source,
            transform: self.transform,
            value_transform: self.value_transform,
            dtype: self.dtype,
            phantom: Phantom::default(),
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let source = self.source.transpose(permutation)?;
        Ok(BlockListUnary {
            source,
            transform: self.transform,
            value_transform: self.value_transform,
            dtype: self.dtype,
            phantom: Phantom::default(),
        })
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(TCError::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(TCError::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>, B: DenseAccess<F, D, T>> ReadValueAt<D>
    for BlockListUnary<F, D, T, B>
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let transform = self.value_transform;
            self.source
                .read_value_at(txn, coord)
                .map_ok(|(coord, value)| (coord, transform(value)))
                .await
        })
    }
}
