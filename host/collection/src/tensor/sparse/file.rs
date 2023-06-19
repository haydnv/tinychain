use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use b_table::b_tree::Collator;
use b_table::{TableLock, TableWriteGuard};
use freqfs::DirLock;
use futures::future::TryFutureExt;
use futures::stream::TryStreamExt;
use ha_ndarray::{ArrayBase, CDatatype};
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{DType, Number, NumberCollator, NumberType};
use tcgeneric::ThreadSafe;

use crate::tensor::{validate_order, Axes, Coord, Range, Shape, TensorInstance};

use super::access::SparseAccess;
use super::schema::{IndexSchema, Schema};
use super::stream::BlockCoords;
use super::{
    table_range, unwrap_row, Elements, Node, SparseInstance, SparseWriteGuard, SparseWriteLock,
};

pub struct SparseFile<FE, T> {
    table: TableLock<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for SparseFile<FE, T> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE, T> SparseFile<FE, T> {
    pub fn collator(&self) -> &Arc<Collator<NumberCollator>> {
        self.table.collator()
    }

    pub(crate) fn into_table(self) -> TableLock<Schema, IndexSchema, NumberCollator, FE> {
        self.table
    }

    pub fn schema(&self) -> &Schema {
        self.table.schema()
    }

    pub(crate) fn table(&self) -> &TableLock<Schema, IndexSchema, NumberCollator, FE> {
        &self.table
    }
}

impl<FE: AsType<Node> + ThreadSafe, T> SparseFile<FE, T> {
    pub async fn copy_from<O>(dir: DirLock<FE>, txn_id: TxnId, other: O) -> TCResult<Self>
    where
        O: SparseInstance<DType = T>,
        T: Into<Number>,
    {
        let this = Self::create(dir, other.shape().clone())?;
        let mut table = this.table.write().await;
        let mut elements = other
            .elements(txn_id, Range::default(), Axes::default())
            .await?;

        while let Some((coord, value)) = elements.try_next().await? {
            let key = coord.into_iter().map(Number::from).collect();
            let value = vec![value.into()];
            table.upsert(key, value).await?;
        }

        Ok(this)
    }

    pub fn create(dir: DirLock<FE>, shape: Shape) -> TCResult<Self> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::create(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }

    pub fn load(dir: DirLock<FE>, shape: Shape) -> TCResult<Self> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::load(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for SparseFile<FE, T>
where
    FE: ThreadSafe,
    T: DType + ThreadSafe,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.table.schema().shape()
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseFile<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType,
    Number: CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = BlockCoords<Elements<T>, T>;
    type DType = T;

    async fn blocks(
        self,
        txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(txn_id, range, order).await?;
        Ok(BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        _txn_id: TxnId,
        range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        self.shape().validate_range(&range)?;
        debug_assert!(validate_order(&order, self.ndim()));

        let range = table_range(&range)?;
        let rows = self.table.rows(range, &order, false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(TCError::from);
        Ok(Box::pin(elements))
    }

    async fn read_value(&self, _txn_id: TxnId, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;

        let key = coord.into_iter().map(Number::from).collect();
        let table = self.table.read().await;
        if let Some(mut row) = table.get(&key).await? {
            let value = row.pop().expect("value");
            Ok(value.cast_into())
        } else {
            Ok(T::zero())
        }
    }
}

#[async_trait]
impl<'a, FE, T> SparseWriteLock<'a> for SparseFile<FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Guard = SparseFileWriteGuard<'a, FE, T>;

    async fn write(&'a self) -> SparseFileWriteGuard<'a, FE, T> {
        SparseFileWriteGuard {
            shape: self.table.schema().shape(),
            table: self.table.write().await,
            dtype: self.dtype,
        }
    }
}

impl<Txn, FE, T: CDatatype> From<SparseFile<FE, T>> for SparseAccess<Txn, FE, T> {
    fn from(table: SparseFile<FE, T>) -> Self {
        Self::Table(table)
    }
}

impl<FE, T> fmt::Debug for SparseFile<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "sparse table with shape {:?}",
            self.table.schema().shape()
        )
    }
}

pub struct SparseFileWriteGuard<'a, FE, T> {
    shape: &'a Shape,
    table: TableWriteGuard<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

#[async_trait]
impl<'a, FE, T> SparseWriteGuard<T> for SparseFileWriteGuard<'a, FE, T>
where
    FE: AsType<Node> + ThreadSafe,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T>,
{
    async fn clear(&mut self, _txn_id: TxnId, range: Range) -> TCResult<()> {
        if range == Range::default() || range == Range::all(&self.shape) {
            self.table.truncate().map_err(TCError::from).await
        } else {
            Err(not_implemented!("delete {range:?}"))
        }
    }

    async fn overwrite<O: SparseInstance<DType = T>>(
        &mut self,
        txn_id: TxnId,
        other: O,
    ) -> TCResult<()> {
        if self.shape != other.shape() {
            return Err(bad_request!(
                "cannot overwrite a tensor of shape {:?} with {:?}",
                self.shape,
                other.shape()
            ));
        }

        self.clear(txn_id, Range::default()).await?;

        let mut elements = other
            .elements(txn_id, Range::default(), Axes::default())
            .await?;

        while let Some((coord, value)) = elements.try_next().await? {
            let coord = coord.into_iter().map(|i| Number::UInt(i.into())).collect();
            self.table.upsert(coord, vec![value.into()]).await?;
        }

        Ok(())
    }

    async fn write_value(&mut self, _txn_id: TxnId, coord: Coord, value: T) -> Result<(), TCError> {
        self.shape.validate_coord(&coord)?;

        let coord = coord.into_iter().map(|i| Number::UInt(i.into())).collect();

        if value == T::zero() {
            self.table.delete(&coord).await?;
        } else {
            self.table.upsert(coord, vec![value.into()]).await?;
        }

        Ok(())
    }
}
