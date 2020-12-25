use async_trait::async_trait;
use futures::future::{self, TryFutureExt};
use futures::stream::{self, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;

use crate::error;
use crate::general::{TCBoxTryFuture, TCResult, TCStream, TCTryStream};
use crate::scalar::number::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::super::sparse::{SparseAccess, SparseTensor};
use super::super::stream::*;
use super::super::transform::{self, Rebase};
use super::super::{
    Bounds, Coord, Shape, TensorAccess, TensorIO, TensorTransform, ERR_NONBIJECTIVE_WRITE,
};
use super::{Array, DenseAccess, DenseAccessor, DenseTensor};

#[derive(Clone)]
pub struct BlockListCombine<L: DenseAccess, R: DenseAccess> {
    left: L,
    right: R,
    combinator: fn(&Array, &Array) -> Array,
    value_combinator: fn(Number, Number) -> Number,
    dtype: NumberType,
}

impl<L: DenseAccess, R: DenseAccess> BlockListCombine<L, R> {
    pub fn new(
        left: L,
        right: R,
        combinator: fn(&Array, &Array) -> Array,
        value_combinator: fn(Number, Number) -> Number,
        dtype: NumberType,
    ) -> TCResult<BlockListCombine<L, R>> {
        if left.shape() != right.shape() {
            return Err(error::bad_request(
                &format!("Cannot combine shape {} with shape", left.shape()),
                right.shape(),
            ));
        }

        Ok(BlockListCombine {
            left,
            right,
            combinator,
            value_combinator,
            dtype,
        })
    }
}

impl<L: DenseAccess, R: DenseAccess> TensorAccess for BlockListCombine<L, R> {
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
impl<L: Clone + DenseAccess, R: Clone + DenseAccess> DenseAccess for BlockListCombine<L, R> {
    type Slice = BlockListCombine<<L as DenseAccess>::Slice, <R as DenseAccess>::Slice>;
    type Transpose = BlockListCombine<<L as DenseAccess>::Transpose, <R as DenseAccess>::Transpose>;

    fn accessor(self) -> DenseAccessor {
        let left = self.left.accessor();
        let right = self.right.accessor();
        let combine = BlockListCombine {
            left,
            right,
            combinator: self.combinator,
            value_combinator: self.value_combinator,
            dtype: self.dtype,
        };

        DenseAccessor::Combine(Box::new(combine))
    }

    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let left = self.left.block_stream(txn);
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
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<L: DenseAccess, R: DenseAccess> ReadValueAt for BlockListCombine<L, R> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let left = self.left.read_value_at(txn, coord.to_vec());
            let right = self.right.read_value_at(txn, coord);
            let ((coord, left), (_, right)) = try_join!(left, right)?;
            let value = (self.value_combinator)(left, right);
            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<L: DenseAccess, R: DenseAccess> Transact for BlockListCombine<L, R> {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

#[derive(Clone)]
pub struct BlockListBroadcast<T: DenseAccess> {
    source: T,
    rebase: transform::Broadcast,
}

impl<T: DenseAccess> BlockListBroadcast<T> {
    pub fn new(source: T, shape: Shape) -> TCResult<Self> {
        let rebase = transform::Broadcast::new(source.shape().clone(), shape)?;
        Ok(Self { source, rebase })
    }
}

impl<T: DenseAccess> TensorAccess for BlockListBroadcast<T> {
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
impl<T: Clone + DenseAccess> DenseAccess for BlockListBroadcast<T> {
    type Slice = BlockListSlice<Self>;
    type Transpose = BlockListTranspose<Self>;

    fn accessor(self) -> DenseAccessor {
        let source = self.source.accessor();
        let broadcast = BlockListBroadcast {
            source,
            rebase: self.rebase,
        };

        DenseAccessor::Broadcast(Box::new(broadcast))
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        let values = stream::iter(Bounds::all(self.shape()).affected())
            .then(move |coord| self.read_value_at(txn, coord).map_ok(|(_, value)| value));

        let values: TCTryStream<'a, Number> = Box::pin(values);
        Box::pin(future::ready(Ok(values)))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        BlockListSlice::new(self, bounds)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        BlockListTranspose::new(self, permutation)
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<T: Clone + DenseAccess> ReadValueAt for BlockListBroadcast<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for BlockListBroadcast<T> {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

#[derive(Clone)]
pub struct BlockListCast<T: DenseAccess> {
    source: T,
    dtype: NumberType,
}

impl<T: DenseAccess> BlockListCast<T> {
    pub fn new(source: T, dtype: NumberType) -> Self {
        Self { source, dtype }
    }
}

impl<T: DenseAccess> TensorAccess for BlockListCast<T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
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
impl<T: Clone + DenseAccess> DenseAccess for BlockListCast<T> {
    type Slice = BlockListCast<<T as DenseAccess>::Slice>;
    type Transpose = BlockListCast<<T as DenseAccess>::Transpose>;

    fn accessor(self) -> DenseAccessor {
        let cast = BlockListCast {
            source: self.source.accessor(),
            dtype: self.dtype,
        };

        DenseAccessor::Cast(Box::new(cast))
    }

    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let dtype = self.dtype;
            let blocks: TCStream<'a, TCResult<Array>> = self.source.block_stream(txn).await?;
            let cast = blocks.map_ok(move |array| array.into_type(dtype));
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

impl<T: Clone + DenseAccess> ReadValueAt for BlockListCast<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let dtype = self.dtype;
        let read = self
            .source
            .read_value_at(txn, coord)
            .map_ok(move |(coord, value)| (coord, value.into_type(dtype)));

        Box::pin(read)
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for BlockListCast<T> {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub struct BlockListExpand<T: DenseAccess> {
    source: T,
    rebase: transform::Expand,
}

impl<T: DenseAccess> BlockListExpand<T> {
    pub fn new(source: T, axis: usize) -> TCResult<Self> {
        let rebase = transform::Expand::new(source.shape().clone(), axis)?;
        Ok(Self { source, rebase })
    }
}

impl<T: DenseAccess> TensorAccess for BlockListExpand<T> {
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
impl<T: DenseAccess> DenseAccess for BlockListExpand<T> {
    type Slice = <T as DenseAccess>::Slice;
    type Transpose = <T as DenseAccess>::Transpose;

    fn accessor(self) -> DenseAccessor {
        let expand = BlockListExpand {
            source: self.source.accessor(),
            rebase: self.rebase,
        };
        DenseAccessor::Expand(Box::new(expand))
    }

    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        self.source.block_stream(txn)
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
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

impl<T: DenseAccess> ReadValueAt for BlockListExpand<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, value)| (coord, value));
        Box::pin(read)
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for BlockListExpand<T> {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

// TODO: &Txn, not Txn
type Reductor = fn(&DenseTensor<DenseAccessor>, Txn) -> TCBoxTryFuture<Number>;

#[derive(Clone)]
pub struct BlockListReduce<T> {
    source: T,
    rebase: transform::Reduce,
    reductor: Reductor,
}

impl<T: DenseAccess> BlockListReduce<T> {
    fn new(source: T, axis: usize, reductor: Reductor) -> TCResult<BlockListReduce<T>> {
        let rebase = transform::Reduce::new(source.shape().clone(), axis)?;

        Ok(BlockListReduce {
            source,
            rebase,
            reductor,
        })
    }
}

impl<T: DenseAccess> TensorAccess for BlockListReduce<T> {
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
impl<T: Clone + DenseAccess> DenseAccess for BlockListReduce<T> {
    type Slice = BlockListReduce<<T as DenseAccess>::Slice>;
    type Transpose = BlockListReduce<<T as DenseAccess>::Transpose>;

    fn accessor(self) -> DenseAccessor {
        let reduce = BlockListReduce {
            source: self.source.accessor(),
            rebase: self.rebase,
            reductor: self.reductor,
        };

        DenseAccessor::Reduce(Box::new(reduce))
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let values = stream::iter(Bounds::all(self.shape()).affected()).then(move |coord| {
                let reductor = self.reductor;
                let source_bounds = self.rebase.invert_coord(&coord);
                Box::pin(async move {
                    let slice = self.source.clone().slice(source_bounds)?;
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
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<T: Clone + DenseAccess> ReadValueAt for BlockListReduce<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let reductor = self.reductor;
            let source_bounds = self.rebase.invert_coord(&coord);
            let slice = self.source.clone().slice(source_bounds)?;
            let value = reductor(&slice.accessor().into(), txn.clone()).await?;

            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for BlockListReduce<T> {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}

#[derive(Clone)]
pub struct BlockListSlice<T> {
    source: T,
    rebase: transform::Slice,
}

impl<T: DenseAccess> BlockListSlice<T> {
    pub fn new(source: T, bounds: Bounds) -> TCResult<Self> {
        let rebase = transform::Slice::new(source.shape().clone(), bounds)?;
        Ok(Self { source, rebase })
    }
}

impl<T: DenseAccess> TensorAccess for BlockListSlice<T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.rebase.ndim()
    }

    fn shape(&'_ self) -> &'_ Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.rebase.size()
    }
}

#[async_trait]
impl<T: Clone + DenseAccess> DenseAccess for BlockListSlice<T> {
    type Slice = <T as DenseAccess>::Slice;
    type Transpose = BlockListTranspose<Self>;

    fn accessor(self) -> DenseAccessor {
        let slice = BlockListSlice {
            source: self.source.accessor(),
            rebase: self.rebase,
        };

        DenseAccessor::Slice(Box::new(slice))
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        let bounds = self.rebase.invert_bounds(Bounds::all(self.shape()));
        let values = stream::iter(bounds.affected()).then(move |coord| {
            self.source
                .read_value_at(txn, coord)
                .map_ok(|(_, value)| value)
        });

        let values: TCTryStream<'a, Number> = Box::pin(values);
        Box::pin(future::ready(Ok(values)))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.slice(bounds)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        BlockListTranspose::new(self, permutation)
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        debug!("BlockListSlice::write_value {} at {}", value, bounds);

        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, value).await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        let coord = self.rebase.invert_coord(&coord);
        self.source.write_value_at(txn_id, coord, value)
    }
}

impl<T: DenseAccess> ReadValueAt for BlockListSlice<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for BlockListSlice<T> {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub struct BlockListSparse<T: Clone + SparseAccess> {
    source: SparseTensor<T>,
}

impl<T: Clone + SparseAccess> BlockListSparse<T> {
    pub fn new(source: SparseTensor<T>) -> Self {
        BlockListSparse { source }
    }
}

impl<T: Clone + SparseAccess> TensorAccess for BlockListSparse<T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
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
impl<T: Clone + SparseAccess> DenseAccess for BlockListSparse<T> {
    type Slice = BlockListSparse<<T as SparseAccess>::Slice>;
    type Transpose = BlockListSparse<<T as SparseAccess>::Transpose>;

    fn accessor(self) -> DenseAccessor {
        let source = self.source.into_inner().accessor().into();
        DenseAccessor::Sparse(Box::new(BlockListSparse { source }))
    }

    fn block_stream<'a>(&'a self, _txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(future::ready(Err(error::not_implemented(
            "BlockListSparse::block_stream",
        ))))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let slice = self.source.slice(bounds)?;
        Ok(BlockListSparse::new(slice))
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let transpose = self.source.transpose(permutation)?;
        Ok(BlockListSparse::new(transpose))
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        self.source
            .clone()
            .write_value(txn_id, bounds, number)
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        self.source.write_value_at(txn_id, coord, value)
    }
}

impl<T: Clone + SparseAccess> ReadValueAt for BlockListSparse<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        self.source.read_value_at(txn, coord)
    }
}

#[async_trait]
impl<T: Clone + SparseAccess> Transact for BlockListSparse<T> {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub struct BlockListTranspose<T: DenseAccess> {
    source: T,
    rebase: transform::Transpose,
}

impl<T: DenseAccess> BlockListTranspose<T> {
    pub fn new(source: T, permutation: Option<Vec<usize>>) -> TCResult<Self> {
        let rebase = transform::Transpose::new(source.shape().clone(), permutation)?;
        Ok(BlockListTranspose { source, rebase })
    }
}

impl<T: DenseAccess> TensorAccess for BlockListTranspose<T> {
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
impl<T: Clone + DenseAccess> DenseAccess for BlockListTranspose<T> {
    type Slice = BlockListTranspose<<T as DenseAccess>::Slice>;
    type Transpose = Self;

    fn accessor(self) -> DenseAccessor {
        let accessor = BlockListTranspose {
            source: self.source.accessor(),
            rebase: self.rebase,
        };
        DenseAccessor::Transpose(Box::new(accessor))
    }

    fn value_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Number>> {
        Box::pin(async move {
            let coords = stream::iter(Bounds::all(self.shape()).affected().map(TCResult::Ok));
            let values: TCTryStream<'a, (Coord, Number)> =
                Box::pin(ValueReader::new(coords, txn, self));

            let values: TCTryStream<'a, Number> = Box::pin(values.map_ok(|(_, value)| value));
            Ok(values)
        })
    }

    fn slice(self, _bounds: Bounds) -> TCResult<Self::Slice> {
        Err(error::not_implemented("BlockListTranspose::slice"))
    }

    fn transpose(self, _permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        Err(error::not_implemented("BlockListTranspose::transpose"))
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

impl<T: DenseAccess> ReadValueAt for BlockListTranspose<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        let source_coord = self.rebase.invert_coord(&coord);
        let read = self
            .source
            .read_value_at(txn, source_coord)
            .map_ok(|(_, val)| (coord, val));
        Box::pin(read)
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for BlockListTranspose<T> {
    async fn commit(&self, txn_id: &TxnId) {
        self.source.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.source.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.source.finalize(txn_id).await
    }
}

#[derive(Clone)]
pub struct BlockListUnary<T: DenseAccess> {
    source: T,
    transform: fn(&Array) -> Array,
    value_transform: fn(Number) -> Number,
    dtype: NumberType,
}

impl<T: DenseAccess> BlockListUnary<T> {
    pub fn new(
        source: T,
        transform: fn(&Array) -> Array,
        value_transform: fn(Number) -> Number,
        dtype: NumberType,
    ) -> Self {
        Self {
            source,
            transform,
            value_transform,
            dtype,
        }
    }
}

impl<T: DenseAccess> TensorAccess for BlockListUnary<T> {
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
impl<T: Clone + DenseAccess> DenseAccess for BlockListUnary<T> {
    type Slice = BlockListUnary<<T as DenseAccess>::Slice>;
    type Transpose = BlockListUnary<<T as DenseAccess>::Transpose>;

    fn accessor(self) -> DenseAccessor {
        let unary = BlockListUnary {
            source: self.source.accessor(),
            transform: self.transform,
            value_transform: self.value_transform,
            dtype: self.dtype,
        };
        DenseAccessor::Unary(Box::new(unary))
    }

    fn block_stream<'a>(&'a self, txn: &'a Txn) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
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
        })
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        let source = self.source.transpose(permutation)?;

        Ok(BlockListUnary {
            source,
            transform: self.transform,
            value_transform: self.value_transform,
            dtype: self.dtype,
        })
    }

    async fn write_value(&self, _txn_id: TxnId, _bounds: Bounds, _number: Number) -> TCResult<()> {
        Err(error::unsupported(ERR_NONBIJECTIVE_WRITE))
    }

    fn write_value_at(&self, _txn_id: TxnId, _coord: Coord, _value: Number) -> TCBoxTryFuture<()> {
        Box::pin(future::ready(Err(error::unsupported(
            ERR_NONBIJECTIVE_WRITE,
        ))))
    }
}

impl<T: DenseAccess> ReadValueAt for BlockListUnary<T> {
    fn read_value_at<'a>(&'a self, txn: &'a Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            let transform = self.value_transform;
            self.source
                .read_value_at(txn, coord)
                .map_ok(|(coord, value)| (coord, transform(value)))
                .await
        })
    }
}

#[async_trait]
impl<T: DenseAccess> Transact for BlockListUnary<T> {
    async fn commit(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn rollback(&self, _txn_id: &TxnId) {
        // no-op
    }

    async fn finalize(&self, _txn_id: &TxnId) {
        // no-op
    }
}
