use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use destream::{de, en};
use futures::future::{self, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{CopyFrom, Dir, File, Hash, Persist};
use tc_transact::{IntoView, Transaction, TxnId};
use tc_value::{Number, NumberType, ValueType};
use tcgeneric::{NativeClass, TCTryStream};

use super::{Coord, Schema, Shape, TensorAccess, TensorIO};

use crate::Bounds;
pub use access::{SparseAccess, SparseAccessor};
pub use table::SparseTable;

mod access;
mod table;

pub type SparseRow = (Coord, Number);
pub type SparseStream<'a> = Pin<Box<dyn Stream<Item = TCResult<SparseRow>> + Send + Unpin + 'a>>;

#[derive(Clone)]
pub struct SparseTensor<F: File<Node>, D: Dir, T: Transaction<D>, A: SparseAccess<F, D, T>> {
    accessor: A,
    file: PhantomData<F>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F: File<Node>, D: Dir, T: Transaction<D>, A: SparseAccess<F, D, T>> SparseTensor<F, D, T, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<F, D, T, A> TensorAccess for SparseTensor<F, D, T, A>
where
    F: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<F, D, T>,
{
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn ndim(&self) -> usize {
        self.accessor.ndim()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }

    fn size(&self) -> u64 {
        self.accessor.size()
    }
}

#[async_trait]
impl<F, D, T, A> TensorIO<D> for SparseTensor<F, D, T, A>
where
    F: File<Node>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<F, D, T>,
{
    type Txn = T;

    async fn read_value(&self, txn: Self::Txn, coord: Coord) -> TCResult<Number> {
        self.accessor
            .clone()
            .read_value_at(txn, coord)
            .map_ok(|(_, value)| value)
            .await
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        stream::iter(bounds.affected())
            .map(|coord| self.accessor.write_value(txn_id, coord, value))
            .buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.accessor.write_value(txn_id, coord, value).await
    }
}

#[async_trait]
impl<F, D, T, A> CopyFrom<D, SparseTensor<F, D, T, A>>
    for SparseTensor<F, D, T, SparseTable<F, D, T>>
where
    F: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    A: SparseAccess<F, D, T>,
    D::FileClass: From<BTreeType>,
{
    async fn copy_from(
        instance: SparseTensor<F, D, T, A>,
        store: Self::Store,
        txn: Self::Txn,
    ) -> TCResult<Self> {
        SparseTable::copy_from(instance, store, txn)
            .map_ok(Self::from)
            .await
    }
}

#[async_trait]
impl<'en, F: File<Node>, D: Dir, T: Transaction<D>, A: SparseAccess<F, D, T>> Hash<'en, D>
    for SparseTensor<F, D, T, A>
{
    type Item = SparseRow;
    type Txn = T;

    async fn hashable(&'en self, txn: &'en Self::Txn) -> TCResult<TCTryStream<'en, SparseRow>> {
        self.accessor.clone().filled(txn.clone()).await
    }
}

#[async_trait]
impl<F, D, T> Persist<D> for SparseTensor<F, D, T, SparseTable<F, D, T>>
where
    F: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType>,
{
    type Schema = Schema;
    type Store = D;
    type Txn = T;

    fn schema(&self) -> &Self::Schema {
        self.accessor.schema()
    }

    async fn load(txn: &Self::Txn, schema: Self::Schema, store: Self::Store) -> TCResult<Self> {
        SparseTable::load(txn, schema, store)
            .map_ok(Self::from)
            .await
    }
}

impl<F: File<Node>, D: Dir, T: Transaction<D>, A: SparseAccess<F, D, T>> From<A>
    for SparseTensor<F, D, T, A>
{
    fn from(accessor: A) -> Self {
        Self {
            accessor,
            file: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

#[async_trait]
impl<'en, F: File<Node>, D: Dir, T: Transaction<D>, A: SparseAccess<F, D, T>> IntoView<'en, D>
    for SparseTensor<F, D, T, A>
{
    type Txn = T;
    type View = SparseTensorView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        Ok(SparseTensorView {
            shape: self.shape().to_vec(),
            dtype: self.dtype().into(),
            filled: self.accessor.filled(txn).await?,
        })
    }
}

#[async_trait]
impl<F, D, T> de::FromStream for SparseTensor<F, D, T, SparseTable<F, D, T>>
where
    F: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<BTreeType>,
{
    type Context = T;

    async fn from_stream<De: de::Decoder>(txn: T, decoder: &mut De) -> Result<Self, De::Error> {
        decoder.decode_seq(SparseTensorVisitor::new(txn)).await
    }
}

struct SparseTensorVisitor<F: File<Node>, D: Dir, T: Transaction<D>> {
    txn: T,
    file: PhantomData<F>,
    dir: PhantomData<D>,
}

impl<F: File<Node>, D: Dir, T: Transaction<D>> SparseTensorVisitor<F, D, T> {
    fn new(txn: T) -> Self {
        Self {
            txn,
            file: PhantomData,
            dir: PhantomData,
        }
    }
}

#[async_trait]
impl<F: File<Node>, D: Dir, T: Transaction<D>> de::Visitor for SparseTensorVisitor<F, D, T>
where
    F: TryFrom<D::File, Error = TCError>,
    D::FileClass: From<BTreeType>,
{
    type Value = SparseTensor<F, D, T, SparseTable<F, D, T>>;

    fn expecting() -> &'static str {
        "a SparseTensor"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let schema = seq.next_element::<(Vec<u64>, ValueType)>(()).await?;
        let (shape, dtype) = schema.ok_or_else(|| de::Error::invalid_length(0, "tensor schema"))?;
        let shape = Shape::from(shape);
        let dtype = dtype.try_into().map_err(de::Error::custom)?;

        let txn_id = *self.txn.id();
        let table = SparseTable::create(self.txn.context(), txn_id, (shape, dtype))
            .map_err(de::Error::custom)
            .await?;

        if let Some(table) = seq
            .next_element::<SparseTable<F, D, T>>((table.clone(), txn_id))
            .await?
        {
            Ok(SparseTensor::from(table))
        } else {
            Ok(SparseTensor::from(table))
        }
    }
}

pub struct SparseTensorView<'en> {
    shape: Vec<u64>,
    dtype: ValueType,
    filled: SparseStream<'en>,
}

impl<'en> en::IntoStream<'en> for SparseTensorView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        let schema = (self.shape.to_vec(), self.dtype.path());
        let filled = en::SeqStream::from(self.filled);
        (schema, filled).into_stream(encoder)
    }
}
