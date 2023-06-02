use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;
use std::sync::Arc;

use async_trait::async_trait;
use b_table::b_tree::Collator;
use b_table::{TableLock, TableWriteGuard};
use freqfs::DirLock;
use futures::future::TryFutureExt;
use futures::stream::{Stream, StreamExt, TryStreamExt};
use futures::try_join;
use ha_ndarray::*;
use itertools::Itertools;
use rayon::prelude::*;
use safecast::{AsType, CastInto};

use tc_error::*;
use tc_value::{DType, Number, NumberCollator, NumberType};

use crate::tensor::{
    offset_of, strides_for, validate_order, validate_transpose, Axes, AxisRange, Coord, Range,
    Shape, Strides, TensorInstance,
};

use super::schema::{IndexSchema, Schema};
use super::{stream, Blocks, Elements, Node, SparseInstance};

#[async_trait]
pub trait SparseWrite<'a>: SparseInstance {
    type Guard: SparseWriteGuard<Self::DType>;

    async fn write(&'a self) -> Self::Guard;
}

#[async_trait]
pub trait SparseWriteGuard<T: CDatatype + DType>: Send + Sync {
    async fn merge<FE>(
        &mut self,
        filled: SparseVersion<FE, T>,
        zeros: SparseVersion<FE, T>,
    ) -> Result<(), TCError>
    where
        FE: AsType<Node> + Send + Sync + 'static,
        Number: CastInto<T>,
    {
        let mut zeros = zeros
            .table
            .rows(b_table::Range::default(), &[], false)
            .await?;

        while let Some(row) = zeros.try_next().await? {
            let (coord, zero) = unwrap_row(row);
            self.write_value(coord, zero).await?;
        }

        let mut filled = filled
            .table
            .rows(b_table::Range::default(), &[], false)
            .await?;

        while let Some(row) = filled.try_next().await? {
            let (coord, value) = unwrap_row(row);
            self.write_value(coord, value).await?;
        }

        Ok(())
    }

    async fn write_value(&mut self, coord: Coord, value: T) -> Result<(), TCError>;
}

pub enum SparseAccess<FE, T> {
    Table(SparseVersion<FE, T>),
    Broadcast(Box<SparseBroadcast<FE, T>>),
    BroadcastAxis(Box<SparseBroadcastAxis<Self>>),
    Cast(Box<SparseCast<FE, T>>),
    Cow(Box<SparseCow<FE, T, Self>>),
    Expand(Box<SparseExpand<Self>>),
    Reshape(Box<SparseReshape<Self>>),
    Slice(Box<SparseSlice<Self>>),
    Transpose(Box<SparseTranspose<Self>>),
}

impl<FE, T> Clone for SparseAccess<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Table(table) => Self::Table(table.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::BroadcastAxis(broadcast) => Self::BroadcastAxis(broadcast.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
            Self::Cow(cow) => Self::Cow(cow.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
        }
    }
}

macro_rules! array_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::Table($var) => $call,
            Self::Broadcast($var) => $call,
            Self::BroadcastAxis($var) => $call,
            Self::Cast($var) => $call,
            Self::Cow($var) => $call,
            Self::Expand($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
            Self::Transpose($var) => $call,
        }
    };
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseAccess<FE, T> {
    fn dtype(&self) -> NumberType {
        array_dispatch!(self, this, this.dtype())
    }

    fn shape(&self) -> &Shape {
        array_dispatch!(self, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseAccess<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Array<u64>, Array<T>>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        match self {
            Self::Table(table) => {
                let blocks = table.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Broadcast(broadcast) => {
                let blocks = broadcast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::BroadcastAxis(broadcast) => {
                let blocks = broadcast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Cast(cast) => {
                let blocks = cast.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Cow(cow) => {
                let blocks = cow.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Expand(expand) => {
                let blocks = expand.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Reshape(reshape) => {
                let blocks = reshape.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Slice(slice) => {
                let blocks = slice.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
            Self::Transpose(transpose) => {
                let blocks = transpose.blocks(range, order).await?;
                let blocks =
                    blocks.map_ok(|(coords, values)| (Array::from(coords), Array::from(values)));

                Ok(Box::pin(blocks))
            }
        }
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        array_dispatch!(self, this, this.elements(range, order).await)
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        array_dispatch!(self, this, this.read_value(coord).await)
    }
}

impl<FE, T: DType> fmt::Debug for SparseAccess<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        array_dispatch!(self, this, this.fmt(f))
    }
}

pub struct SparseVersion<FE, T> {
    table: TableLock<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for SparseVersion<FE, T> {
    fn clone(&self) -> Self {
        Self {
            table: self.table.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE, T> SparseVersion<FE, T> {
    pub fn collator(&self) -> &Arc<Collator<NumberCollator>> {
        self.table.collator()
    }

    pub fn schema(&self) -> &Schema {
        self.table.schema()
    }
}

impl<FE: AsType<Node> + Send + Sync, T> SparseVersion<FE, T> {
    pub fn create(dir: DirLock<FE>, shape: Shape) -> Result<Self, TCError> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::create(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }

    pub fn load(dir: DirLock<FE>, shape: Shape) -> Result<Self, TCError> {
        let schema = Schema::new(shape);
        let collator = NumberCollator::default();
        let table = TableLock::load(schema, collator, dir)?;

        Ok(Self {
            table,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for SparseVersion<FE, T>
where
    FE: Send + Sync + 'static,
    T: DType + Send + Sync + 'static,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.table.schema().shape()
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseVersion<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType,
    Number: CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<T>, T>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        debug_assert!(self.shape().validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let range = table_range(&range)?;
        let rows = self.table.rows(range, &order, false).await?;
        let elements = rows.map_ok(|row| unwrap_row(row)).map_err(TCError::from);
        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
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
impl<'a, FE, T> SparseWrite<'a> for SparseVersion<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Guard = SparseTableWriteGuard<'a, FE, T>;

    async fn write(&'a self) -> SparseTableWriteGuard<'a, FE, T> {
        SparseTableWriteGuard {
            shape: self.table.schema().shape(),
            table: self.table.write().await,
            dtype: self.dtype,
        }
    }
}

impl<FE, T> From<SparseVersion<FE, T>> for SparseAccess<FE, T> {
    fn from(table: SparseVersion<FE, T>) -> Self {
        Self::Table(table)
    }
}

impl<FE, T> fmt::Debug for SparseVersion<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "sparse table with shape {:?}",
            self.table.schema().shape()
        )
    }
}

pub struct SparseTableWriteGuard<'a, FE, T> {
    shape: &'a Shape,
    table: TableWriteGuard<Schema, IndexSchema, NumberCollator, FE>,
    dtype: PhantomData<T>,
}

#[async_trait]
impl<'a, FE, T> SparseWriteGuard<T> for SparseTableWriteGuard<'a, FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T>,
{
    async fn write_value(&mut self, coord: Coord, value: T) -> Result<(), TCError> {
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

pub struct SparseBroadcast<FE, T> {
    shape: Shape,
    inner: SparseAccess<FE, T>,
}

impl<FE, T> Clone for SparseBroadcast<FE, T> {
    fn clone(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> SparseBroadcast<FE, T> {
    pub fn new<S>(source: S, shape: Shape) -> Result<Self, TCError>
    where
        S: TensorInstance + Into<SparseAccess<FE, T>>,
    {
        let source_shape = source.shape().to_vec();
        let mut inner = source.into();

        let axes = (0..source_shape.len()).into_iter().rev();
        let dims = source_shape
            .into_iter()
            .rev()
            .zip(shape.iter().rev().copied());

        for (x, (dim, bdim)) in axes.zip(dims) {
            if dim == bdim {
                // no-op
            } else if dim == 1 {
                let broadcast_axis = SparseBroadcastAxis::new(inner, x, bdim)?;
                inner = SparseAccess::BroadcastAxis(Box::new(broadcast_axis));
            } else {
                return Err(bad_request!(
                    "cannot broadcast {} into {} at axis {}",
                    dim,
                    bdim,
                    x
                ));
            }
        }

        Ok(Self { shape, inner })
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseBroadcast<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseBroadcast<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        mut range: Range,
        order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let offset = ndim - self.inner.ndim();

        if offset == 0 {
            return self.inner.elements(range, order).await;
        }

        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, ndim));

        let mut inner_range = Vec::with_capacity(self.inner.ndim());
        while range.len() > offset {
            inner_range.push(range.pop().expect("bound"));
        }

        let inner_order = if order
            .iter()
            .take(offset)
            .copied()
            .enumerate()
            .all(|(o, x)| x == o)
        {
            Ok(order.iter().skip(offset).cloned().collect::<Axes>())
        } else {
            Err(bad_request!(
                "an outer broadcast of a sparse tensor does not support permutation"
            ))
        }?;

        let outer = range.into_iter().multi_cartesian_product();

        let inner = self.inner;
        let elements = futures::stream::iter(outer)
            .then(move |outer_coord| {
                let inner = inner.clone();
                let inner_range = inner_range.to_vec();
                let inner_order = inner_order.to_vec();

                async move {
                    let inner_elements = inner.elements(inner_range.into(), inner_order).await?;

                    let elements = inner_elements.map_ok(move |(inner_coord, value)| {
                        let mut coord = Vec::with_capacity(ndim);
                        coord.extend_from_slice(&outer_coord);
                        coord.extend(inner_coord);
                        (coord, value)
                    });

                    Result::<_, TCError>::Ok(elements)
                }
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, mut coord: Coord) -> Result<Self::DType, TCError> {
        while coord.len() > self.inner.ndim() {
            coord.remove(0);
        }

        self.inner.read_value(coord).await
    }
}

impl<FE, T> From<SparseBroadcast<FE, T>> for SparseAccess<FE, T> {
    fn from(accessor: SparseBroadcast<FE, T>) -> Self {
        Self::Broadcast(Box::new(accessor))
    }
}

impl<FE, T> fmt::Debug for SparseBroadcast<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "broadcasted sparse tensor with shape {:?}", self.shape)
    }
}

#[derive(Clone)]
pub struct SparseBroadcastAxis<S> {
    source: S,
    axis: usize,
    dim: u64,
    shape: Shape,
}

impl<S: TensorInstance + fmt::Debug> SparseBroadcastAxis<S> {
    fn new(source: S, axis: usize, dim: u64) -> Result<Self, TCError> {
        let shape = if axis < source.ndim() {
            let mut shape = source.shape().to_vec();
            if shape[axis] == 1 {
                shape[axis] = dim;
                Ok(shape)
            } else {
                Err(bad_request!(
                    "cannot broadcast dimension {} into {}",
                    shape[axis],
                    dim
                ))
            }
        } else {
            Err(bad_request!("invalid axis for {:?}: {}", source, axis))
        }?;

        Ok(Self {
            source,
            axis,
            dim,
            shape: shape.into(),
        })
    }
}

impl<S: TensorInstance> TensorInstance for SparseBroadcastAxis<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance + Clone> SparseInstance for SparseBroadcastAxis<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(
        self,
        range: Range,
        mut order: Axes,
    ) -> Result<Elements<Self::DType>, TCError> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let axis = self.axis;
        let ndim = self.shape.len();

        let (source_range, dim) = if range.len() > axis {
            let bdim = match &range[axis] {
                AxisRange::At(i) if *i < self.dim => Ok(1),
                AxisRange::In(axis_range, 1) if axis_range.end > axis_range.start => {
                    Ok(axis_range.end - axis_range.start)
                }
                bound => Err(bad_request!(
                    "invalid bound for axis {}: {:?}",
                    self.axis,
                    bound
                )),
            }?;

            let mut source_range = range;
            source_range[axis] = AxisRange::At(0);
            (source_range, bdim)
        } else {
            (range, self.dim)
        };

        let (source_order, inner_order) = if order
            .iter()
            .take(axis)
            .copied()
            .enumerate()
            .all(|(o, x)| x == o)
        {
            let mut inner_order = Axes::with_capacity(ndim - axis);

            while order.len() > axis {
                inner_order.push(order.pop().expect("axis"));
            }

            Ok((order, inner_order))
        } else {
            Err(bad_request!(
                "an outer broadcast of a sparse tensor does not support permutation"
            ))
        }?;

        if self.axis == self.ndim() - 1 {
            let source_elements = self.source.elements(source_range, source_order).await?;

            // TODO: write a range to a slice of a coordinate block instead
            let elements = source_elements
                .map_ok(move |(source_coord, value)| {
                    futures::stream::iter(0..dim).map(move |i| {
                        let mut coord = source_coord.to_vec();
                        *coord.last_mut().expect("x") = i;

                        Ok((coord, value))
                    })
                })
                .try_flatten();

            Ok(Box::pin(elements))
        } else {
            let axes = (0..axis).into_iter().collect();
            let inner_range = source_range.iter().skip(axis).cloned().collect::<Vec<_>>();

            let source = self.source;
            let filled = source.clone().filled_at(source_range, axes).await?;

            let elements = filled
                .map(move |result| {
                    let outer_coord = result?;
                    debug_assert_eq!(outer_coord.len(), axis);
                    debug_assert_eq!(outer_coord.last(), Some(&0));

                    let inner_range = inner_range.to_vec();
                    let inner_order = inner_order.to_vec();

                    let prefix = outer_coord
                        .iter()
                        .copied()
                        .map(|i| AxisRange::At(i))
                        .collect();

                    let slice = SparseSlice::new(source.clone(), prefix)?;

                    let elements = futures::stream::iter(0..dim)
                        .then(move |i| {
                            let outer_coord = outer_coord[..axis - 1].to_vec();
                            let inner_range = inner_range.to_vec().into();
                            let inner_order = inner_order.to_vec();
                            let slice = slice.clone();

                            async move {
                                let inner_elements =
                                    slice.elements(inner_range, inner_order).await?;

                                let elements =
                                    inner_elements.map_ok(move |(inner_coord, value)| {
                                        let mut coord = Coord::with_capacity(ndim);
                                        coord.copy_from_slice(&outer_coord);
                                        coord.push(i);
                                        coord.extend(inner_coord);

                                        (coord, value)
                                    });

                                Result::<_, TCError>::Ok(elements)
                            }
                        })
                        .try_flatten();

                    Result::<_, TCError>::Ok(elements)
                })
                .try_flatten();

            Ok(Box::pin(elements))
        }
    }

    async fn read_value(&self, mut coord: Coord) -> Result<Self::DType, TCError> {
        self.shape.validate_coord(&coord)?;
        coord[self.axis] = 0;
        self.source.read_value(coord).await
    }
}

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseBroadcastAxis<S>> for SparseAccess<FE, T> {
    fn from(broadcast: SparseBroadcastAxis<S>) -> Self {
        Self::BroadcastAxis(Box::new(SparseBroadcastAxis {
            source: broadcast.source.into(),
            axis: broadcast.axis,
            dim: broadcast.dim,
            shape: broadcast.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseBroadcastAxis<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "broadcast of {:?} axis {}", self.source, self.axis)
    }
}

pub enum SparseCastSource<FE> {
    F32(SparseAccess<FE, f32>),
    F64(SparseAccess<FE, f64>),
    I16(SparseAccess<FE, i16>),
    I32(SparseAccess<FE, i32>),
    I64(SparseAccess<FE, i64>),
    U8(SparseAccess<FE, u8>),
    U16(SparseAccess<FE, u16>),
    U32(SparseAccess<FE, u32>),
    U64(SparseAccess<FE, u64>),
}

impl<FE> Clone for SparseCastSource<FE> {
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

macro_rules! cast_source {
    ($t:ty, $var:ident) => {
        impl<FE> From<SparseAccess<FE, $t>> for SparseCastSource<FE> {
            fn from(access: SparseAccess<FE, $t>) -> Self {
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
            SparseCastSource::F32($var) => $call,
            SparseCastSource::F64($var) => $call,
            SparseCastSource::I16($var) => $call,
            SparseCastSource::I32($var) => $call,
            SparseCastSource::I64($var) => $call,
            SparseCastSource::U8($var) => $call,
            SparseCastSource::U16($var) => $call,
            SparseCastSource::U32($var) => $call,
            SparseCastSource::U64($var) => $call,
        }
    };
}

impl<FE> fmt::Debug for SparseCastSource<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        cast_dispatch!(self, this, this.fmt(f))
    }
}

pub struct SparseCast<FE, T> {
    source: SparseCastSource<FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for SparseCast<FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            dtype: self.dtype,
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for SparseCast<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let source = &self.source;
        cast_dispatch!(source, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> SparseInstance for SparseCast<FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type CoordBlock = Array<u64>;
    type ValueBlock = Array<T>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        match self.source {
            SparseCastSource::F32(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::F64(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::I16(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::I32(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::I64(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U8(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U16(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U32(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
            SparseCastSource::U64(access) => {
                let source_blocks = access.blocks(range, order).await?;
                let blocks = source_blocks.map(|result| {
                    result.and_then(|(coords, values)| {
                        let values = values.cast().map(Array::<T>::from)?;
                        Ok((coords, values))
                    })
                });

                Ok(Box::pin(blocks))
            }
        }
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.read(&queue)?.to_slice()?;
                let values = values.read(&queue)?.to_slice()?;
                let tuples = coords
                    .as_ref()
                    .into_par_iter()
                    .copied()
                    .chunks(ndim)
                    .zip(values.as_ref().into_par_iter().copied())
                    .map(Ok)
                    .collect::<Vec<_>>();

                Result::<_, TCError>::Ok(futures::stream::iter(tuples))
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        let source = &self.source;
        cast_dispatch!(source, this, {
            let value: Number = this.read_value(coord).map_ok(|n| n.into()).await?;
            Ok(value.cast_into())
        })
    }
}

impl<FE, T: DType> fmt::Debug for SparseCast<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cast {:?} into {:?}", self.source, T::dtype())
    }
}

pub struct SparseCow<FE, T, S> {
    source: S,
    filled: SparseVersion<FE, T>,
    zeros: SparseVersion<FE, T>,
}

impl<FE, T, S: Clone> Clone for SparseCow<FE, T, S> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            filled: self.filled.clone(),
            zeros: self.zeros.clone(),
        }
    }
}

impl<FE, T, S> SparseCow<FE, T, S> {
    pub fn create(source: S, filled: SparseVersion<FE, T>, zeros: SparseVersion<FE, T>) -> Self {
        Self {
            source,
            filled,
            zeros,
        }
    }

    pub fn into_deltas(self) -> (SparseVersion<FE, T>, SparseVersion<FE, T>) {
        (self.filled, self.zeros)
    }
}

impl<FE, T, S> TensorInstance for SparseCow<FE, T, S>
where
    FE: Send + Sync + 'static,
    T: CDatatype + DType,
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<FE, T, S> SparseInstance for SparseCow<FE, T, S>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    S: SparseInstance<DType = T>,
    Number: CastInto<T>,
{
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<T>>;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = T;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let shape = self.source.shape().to_vec();
        let ndim = shape.len();
        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context.clone(), size_hint(self.size()))?;

        let strides = strides_for(&shape, ndim);
        let strides = ArrayBase::<Arc<Vec<_>>>::with_context(
            context.clone(),
            vec![strides.len()],
            Arc::new(strides),
        )?;

        let (source_blocks, filled_blocks, zero_blocks) = try_join!(
            self.source.blocks(range.clone(), order.to_vec()),
            self.filled.blocks(range.clone(), order.to_vec()),
            self.zeros.blocks(range, order)
        )?;

        let source_elements = offsets(queue.clone(), strides.clone(), source_blocks);
        let filled_elements = offsets(queue.clone(), strides.clone(), filled_blocks);
        let zero_elements = offsets(queue, strides.clone(), zero_blocks);

        let elements = stream::TryDiff::new(source_elements, zero_elements);
        let elements = stream::TryMerge::new(elements, filled_elements);
        let offsets = stream::BlockOffsets::new(elements);

        let dims = ArrayBase::<Arc<Vec<_>>>::with_context(context, vec![ndim], Arc::new(shape))?;
        let blocks = offsets.map(move |result| {
            let (offsets, values) = result?;

            let num_offsets = offsets.size();
            let block_shape = vec![num_offsets, ndim];
            let coords = offsets
                .expand_dims(vec![1])?
                .broadcast(block_shape.to_vec())?
                .mul(strides.clone().broadcast(block_shape.to_vec())?)?
                .rem(dims.clone().broadcast(block_shape)?)?;

            let coords = ArrayBase::<Vec<_>>::copy(&coords)?;

            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();
        let blocks = self.blocks(range, order).await?;
        let elements = blocks
            .map_ok(move |(coords, values)| {
                let tuples = coords
                    .into_inner()
                    .into_par_iter()
                    .chunks(ndim)
                    .zip(values.into_inner());

                futures::stream::iter(tuples.map(Ok).collect::<Vec<_>>())
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape().validate_coord(&coord)?;

        let key = coord.iter().copied().map(Number::from).collect();

        {
            let zeros = self.zeros.table.read().await;
            if zeros.contains(&key).await? {
                return Ok(Self::DType::zero());
            }
        }

        {
            let filled = self.filled.table.read().await;
            if let Some(mut row) = filled.get(&key).await? {
                let value = row.pop().expect("value");
                return Ok(value.cast_into());
            }
        }

        self.source.read_value(coord).await
    }
}

#[async_trait]
impl<'a, FE, T, S> SparseWrite<'a> for SparseCow<FE, T, S>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    S: SparseInstance<DType = T>,
    Number: From<T> + CastInto<T>,
{
    type Guard = SparseCowWriteGuard<'a, FE, T>;

    async fn write(&'a self) -> Self::Guard {
        SparseCowWriteGuard {
            filled: self.filled.write().await,
            zeros: self.zeros.write().await,
        }
    }
}

impl<FE, T, S> From<SparseCow<FE, T, S>> for SparseAccess<FE, T>
where
    S: Into<SparseAccess<FE, T>>,
{
    fn from(cow: SparseCow<FE, T, S>) -> Self {
        SparseAccess::Cow(Box::new(SparseCow {
            source: cow.source.into(),
            filled: cow.filled,
            zeros: cow.zeros,
        }))
    }
}

impl<FE, T, S: fmt::Debug> fmt::Debug for SparseCow<FE, T, S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "copy-on-write view of {:?}", self.source)
    }
}

pub struct SparseCowWriteGuard<'a, FE, T> {
    filled: SparseTableWriteGuard<'a, FE, T>,
    zeros: SparseTableWriteGuard<'a, FE, T>,
}

#[async_trait]
impl<'a, FE, T> SparseWriteGuard<T> for SparseCowWriteGuard<'a, FE, T>
where
    FE: AsType<Node> + Send + Sync + 'static,
    T: CDatatype + DType + fmt::Debug,
    Number: From<T>,
{
    async fn write_value(&mut self, coord: Coord, value: T) -> Result<(), TCError> {
        let inverse = if value == T::zero() {
            T::one()
        } else {
            T::zero()
        };

        try_join!(
            self.filled.write_value(coord.to_vec(), value),
            self.zeros.write_value(coord, inverse)
        )
        .map(|_| ())
    }
}

#[derive(Clone)]
pub struct SparseExpand<S> {
    source: S,
    shape: Shape,
    axes: Axes,
}

impl<S: TensorInstance + fmt::Debug> SparseExpand<S> {
    pub fn new(source: S, mut axes: Axes) -> Result<Self, TCError> {
        axes.sort();

        let mut shape = source.shape().to_vec();
        for x in axes.iter().rev().copied() {
            shape.insert(x, 1);
        }

        if Some(source.ndim()) > axes.last().copied() {
            Ok(Self {
                source,
                shape: shape.into(),
                axes,
            })
        } else {
            Err(bad_request!(
                "cannot expand axes {:?} of {:?}",
                axes,
                source
            ))
        }
    }
}

impl<S: TensorInstance> TensorInstance for SparseExpand<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseExpand<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = ArrayBase<Vec<Self::DType>>;
    type Blocks = stream::BlockCoords<Elements<Self::DType>, Self::DType>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        let ndim = self.ndim();
        let elements = self.elements(range, order).await?;
        Ok(stream::BlockCoords::new(elements, ndim))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let mut source_range = range;
        for x in self.axes.iter().rev().copied() {
            if x < source_range.len() {
                source_range.remove(x);
            }
        }

        let mut source_order = order;
        for x in self.axes.iter().rev().copied() {
            source_order.remove(x);
        }

        let ndim = self.ndim();
        let axes = self.axes;
        debug_assert_eq!(self.source.ndim() + 1, ndim);

        let source_elements = self.source.elements(source_range, source_order).await?;

        let elements = source_elements.map_ok(move |(source_coord, value)| {
            let mut coord = Coord::with_capacity(ndim);
            coord.extend(source_coord);
            for x in axes.iter().rev().copied() {
                coord.insert(x, 0);
            }

            debug_assert_eq!(coord.len(), ndim);
            (coord, value)
        });

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, mut coord: Coord) -> Result<Self::DType, TCError> {
        self.shape.validate_coord(&coord)?;

        for x in self.axes.iter().rev() {
            coord.remove(*x);
        }

        self.source.read_value(coord).await
    }
}

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseExpand<S>> for SparseAccess<FE, T> {
    fn from(expand: SparseExpand<S>) -> Self {
        Self::Expand(Box::new(SparseExpand {
            source: expand.source.into(),
            shape: expand.shape,
            axes: expand.axes,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseExpand<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "expand axes {:?} of {:?}", self.axes, self.source)
    }
}

#[derive(Clone)]
pub struct SparseReshape<S> {
    source: S,
    source_strides: Strides,
    shape: Shape,
    strides: Strides,
}

impl<S: SparseInstance> SparseReshape<S> {
    pub fn new(source: S, shape: Shape) -> Result<Self, TCError> {
        if source.shape().iter().product::<u64>() != shape.iter().product::<u64>() {
            return Err(bad_request!("cannot reshape {:?} into {:?}", source, shape));
        }

        let source_strides = strides_for(source.shape(), source.ndim());
        let strides = strides_for(&shape, shape.len());

        Ok(Self {
            source,
            source_strides,
            shape,
            strides,
        })
    }
}

impl<S: TensorInstance> TensorInstance for SparseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: SparseInstance> SparseInstance for SparseReshape<S> {
    type CoordBlock = ArrayBase<Vec<u64>>;
    type ValueBlock = S::ValueBlock;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let source_range = if range.is_empty() {
            Ok(range)
        } else {
            Err(bad_request!(
                "cannot slice a reshaped sparse tensor (consider making a copy first)"
            ))
        }?;

        let source_order = if order
            .iter()
            .copied()
            .zip(0..self.ndim())
            .all(|(x, o)| x == o)
        {
            Ok(order)
        } else {
            Err(bad_request!(
                "cannot transpose a reshaped sparse tensor (consider making a copy first)"
            ))
        }?;

        let source_ndim = self.source.ndim();
        let source_blocks = self.source.blocks(source_range, source_order).await?;
        let source_strides =
            ArrayBase::<Arc<Vec<_>>>::new(vec![source_ndim], Arc::new(self.source_strides))?;

        let ndim = self.shape.len();
        let strides = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(self.strides))?;
        let shape = ArrayBase::<Arc<Vec<_>>>::new(vec![ndim], Arc::new(self.shape.into()))?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;

            debug_assert_eq!(source_coords.size() % source_ndim, 0);
            debug_assert_eq!(source_coords.size() / source_ndim, values.size());

            let source_strides = source_strides
                .clone()
                .broadcast(vec![values.size(), source_ndim])?;

            let offsets = source_coords.mul(source_strides)?;
            let offsets = offsets.sum_axis(1)?;

            let broadcast = vec![offsets.size(), ndim];
            let strides = strides.clone().broadcast(broadcast.to_vec())?;
            let offsets = offsets
                .expand_dims(vec![1])?
                .broadcast(broadcast.to_vec())?;

            let dims = shape.clone().expand_dims(vec![0])?.broadcast(broadcast)?;
            let coords = (offsets / strides) % dims;

            let coords = ArrayBase::<Vec<_>>::copy(&coords)?;

            Result::<_, TCError>::Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.shape.len();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.into_inner();
                let values = values.read(&queue)?.to_slice()?;
                let tuples = coords
                    .into_par_iter()
                    .chunks(ndim)
                    .zip(values.as_ref().into_par_iter().copied())
                    .map(Ok)
                    .collect::<Vec<_>>();

                Result::<_, TCError>::Ok(futures::stream::iter(tuples))
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape.validate_coord(&coord)?;
        let offset = offset_of(coord, &self.shape);
        let source_coord = self
            .source_strides
            .iter()
            .copied()
            .zip(self.source.shape().iter().copied())
            .map(|(stride, dim)| (offset / stride) % dim)
            .collect();

        self.source.read_value(source_coord).await
    }
}

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseReshape<S>> for SparseAccess<FE, T> {
    fn from(reshape: SparseReshape<S>) -> Self {
        Self::Reshape(Box::new(SparseReshape {
            source: reshape.source.into(),
            source_strides: reshape.source_strides,
            shape: reshape.shape,
            strides: reshape.strides,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseReshape<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "reshape {:?} into {:?}", self.source, self.shape)
    }
}

#[derive(Clone)]
pub struct SparseSlice<S> {
    source: S,
    range: Range,
    shape: Shape,
}

impl<S> SparseSlice<S>
where
    S: TensorInstance + fmt::Debug,
{
    pub fn new(source: S, range: Range) -> Result<Self, TCError> {
        source.shape().validate_range(&range)?;

        let mut shape = Vec::with_capacity(source.ndim());
        for (x, bound) in range.iter().enumerate() {
            match bound {
                AxisRange::At(_) => {} // no-op
                AxisRange::In(axis_range, 1) => {
                    shape.push(axis_range.end - axis_range.start);
                }
                axis_bound => {
                    return Err(bad_request!(
                        "invalid bound for sparse tensor axis {}: {:?}",
                        x,
                        axis_bound
                    ));
                }
            }
        }

        shape.extend_from_slice(&source.shape()[range.len()..]);

        Ok(Self {
            source,
            range,
            shape: shape.into(),
        })
    }

    fn source_range(&self, range: Range) -> Result<Range, TCError> {
        let mut source_range = Vec::with_capacity(self.source.ndim());
        let mut axis = 0;

        for axis_range in self.range.iter() {
            let axis_range = match axis_range {
                AxisRange::At(i) => AxisRange::At(*i),
                AxisRange::In(source_range, source_step) => match &range[axis] {
                    AxisRange::At(i) => {
                        debug_assert!(source_range.start + (i * source_step) < source_range.end);
                        AxisRange::At(source_range.start + (i * source_step))
                    }
                    AxisRange::In(axis_range, step) => {
                        debug_assert!(source_range.start + axis_range.start <= source_range.end);
                        debug_assert!(source_range.start + axis_range.end <= source_range.end);

                        let (source_start, source_end, source_step) = (
                            axis_range.start + source_range.start,
                            axis_range.end + source_range.start,
                            step * source_step,
                        );

                        AxisRange::In(source_start..source_end, source_step)
                    }
                    AxisRange::Of(indices) => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(|i| source_range.start + i)
                            .collect::<Vec<u64>>();
                        debug_assert!(indices.iter().copied().all(|i| i < source_range.end));
                        AxisRange::Of(indices)
                    }
                },
                AxisRange::Of(source_indices) => match &range[axis] {
                    AxisRange::At(i) => AxisRange::At(source_indices[*i as usize]),
                    AxisRange::In(axis_range, step) => {
                        debug_assert!(axis_range.start as usize <= source_indices.len());
                        debug_assert!(axis_range.end as usize <= source_indices.len());

                        let indices = source_indices
                            [(axis_range.start as usize)..(axis_range.end as usize)]
                            .iter()
                            .step_by(*step as usize)
                            .copied()
                            .collect();

                        AxisRange::Of(indices)
                    }
                    AxisRange::Of(indices) => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(|i| source_indices[i as usize])
                            .collect();

                        AxisRange::Of(indices)
                    }
                },
            };

            if !axis_range.is_index() {
                axis += 1;
            }

            source_range.push(axis_range);
        }

        source_range.extend(self.range.iter().skip(range.len()).cloned());

        Ok(source_range.into())
    }

    fn source_order(&self, order: Axes) -> Result<Axes, TCError> {
        debug_assert!(validate_order(&order, self.ndim()));

        let mut source_axes = Vec::with_capacity(self.ndim());
        for (x, bound) in self.range.iter().enumerate() {
            if !bound.is_index() {
                source_axes.push(x);
            }
        }

        Ok(order.into_iter().map(|x| source_axes[x]).collect())
    }
}

impl<S> TensorInstance for SparseSlice<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S> SparseInstance for SparseSlice<S>
where
    S: SparseInstance,
{
    type CoordBlock = S::CoordBlock;
    type ValueBlock = S::ValueBlock;
    type Blocks = S::Blocks;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        debug_assert!(self.shape.validate_range(&range).is_ok());

        let source_order = self.source_order(order)?;

        let source_range = if range.is_empty() {
            self.range
        } else {
            self.source_range(range)?
        };

        self.source.blocks(source_range, source_order).await
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        debug_assert!(self.shape.validate_range(&range).is_ok());

        let source_order = self.source_order(order)?;

        let source_range = if range.is_empty() {
            self.range
        } else {
            self.source_range(range)?
        };

        self.source.elements(source_range, source_order).await
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape.validate_coord(&coord)?;
        let source_coord = self.range.invert_coord(coord)?;
        self.source.read_value(source_coord).await
    }
}

#[async_trait]
impl<'a, S> SparseWrite<'a> for SparseSlice<S>
where
    S: SparseWrite<'a>,
{
    type Guard = SparseSliceWriteGuard<'a, S::Guard, S::DType>;

    async fn write(&'a self) -> SparseSliceWriteGuard<'a, S::Guard, S::DType> {
        SparseSliceWriteGuard {
            shape: &self.shape,
            range: &self.range,
            guard: self.source.write().await,
            dtype: PhantomData,
        }
    }
}

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseSlice<S>> for SparseAccess<FE, T> {
    fn from(slice: SparseSlice<S>) -> Self {
        Self::Slice(Box::new(SparseSlice {
            source: slice.source.into(),
            range: slice.range,
            shape: slice.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "slice of {:?} with range {:?}", self.source, self.range)
    }
}

pub struct SparseSliceWriteGuard<'a, G, T> {
    shape: &'a Shape,
    range: &'a Range,
    guard: G,
    dtype: PhantomData<T>,
}

#[async_trait]
impl<'a, G, T> SparseWriteGuard<T> for SparseSliceWriteGuard<'a, G, T>
where
    G: SparseWriteGuard<T>,
    T: CDatatype + DType,
{
    async fn write_value(&mut self, coord: Coord, value: T) -> Result<(), TCError> {
        self.shape.validate_coord(&coord)?;
        let coord = self.range.invert_coord(coord)?;
        self.guard.write_value(coord, value).await
    }
}

#[derive(Clone)]
pub struct SparseTranspose<S> {
    source: S,
    permutation: Axes,
    shape: Shape,
}

impl<S: SparseInstance> SparseTranspose<S> {
    pub fn new(source: S, permutation: Option<Axes>) -> Result<Self, TCError> {
        let permutation = validate_transpose(permutation, source.shape())?;

        let shape = permutation
            .iter()
            .copied()
            .map(|x| source.shape()[x])
            .collect();

        Ok(Self {
            source,
            permutation,
            shape,
        })
    }
}

impl<S> TensorInstance for SparseTranspose<S>
where
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S> SparseInstance for SparseTranspose<S>
where
    S: SparseInstance,
    <S::CoordBlock as NDArrayTransform>::Transpose: NDArrayRead<DType = u64>,
{
    type CoordBlock = <S::CoordBlock as NDArrayTransform>::Transpose;
    type ValueBlock = S::ValueBlock;
    type Blocks = Blocks<Self::CoordBlock, Self::ValueBlock>;
    type DType = S::DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError> {
        debug_assert!(self.shape.validate_range(&range).is_ok());
        debug_assert!(validate_order(&order, self.ndim()));

        let range = range.normalize(self.shape());
        debug_assert_eq!(range.len(), self.ndim());

        let permutation = self.permutation;
        let mut source_range = Range::all(self.source.shape());
        for axis in 0..range.len() {
            source_range[permutation[axis]] = range[axis].clone();
        }

        let source_order = order.into_iter().map(|x| permutation[x]).collect();

        let source_blocks = self.source.blocks(source_range, source_order).await?;

        let blocks = source_blocks.map(move |result| {
            let (source_coords, values) = result?;
            let coords = source_coords.transpose(Some(permutation.to_vec()))?;
            Ok((coords, values))
        });

        Ok(Box::pin(blocks))
    }

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError> {
        let ndim = self.ndim();

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, size_hint(self.size()))?;

        let blocks = self.blocks(range, order).await?;

        let elements = blocks
            .map(move |result| {
                let (coords, values) = result?;
                let coords = coords.read(&queue)?.to_slice()?;
                let values = values.read(&queue)?.to_slice()?;
                let tuples = coords
                    .as_ref()
                    .into_par_iter()
                    .copied()
                    .chunks(ndim)
                    .zip(values.as_ref().into_par_iter().copied())
                    .map(Ok)
                    .collect::<Vec<_>>();

                Result::<_, TCError>::Ok(futures::stream::iter(tuples))
            })
            .try_flatten();

        Ok(Box::pin(elements))
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError> {
        self.shape.validate_coord(&coord)?;

        let mut source_coord = vec![0; coord.len()];
        for (i, x) in self.permutation.iter().copied().enumerate() {
            source_coord[x] = coord[i];
        }

        self.source.read_value(source_coord).await
    }
}

impl<FE, T, S: Into<SparseAccess<FE, T>>> From<SparseTranspose<S>> for SparseAccess<FE, T> {
    fn from(transpose: SparseTranspose<S>) -> Self {
        Self::Transpose(Box::new(SparseTranspose {
            source: transpose.source.into(),
            permutation: transpose.permutation,
            shape: transpose.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for SparseTranspose<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "transpose of {:?} with permutation {:?}",
            self.source, self.permutation
        )
    }
}

#[inline]
fn offsets<C, V, T>(
    queue: ha_ndarray::Queue,
    strides: ArrayBase<Arc<Vec<u64>>>,
    blocks: impl Stream<Item = Result<(C, V), TCError>> + Send + 'static,
) -> impl Stream<Item = Result<(u64, T), TCError>> + Send
where
    C: NDArrayRead<DType = u64> + NDArrayMath,
    V: NDArrayRead<DType = T>,
    T: CDatatype,
{
    let offsets = blocks
        .map(move |result| {
            let (coords, values) = result?;

            let strides = strides.clone().broadcast(coords.shape().to_vec())?;
            let offsets = coords.mul(strides)?.sum_axis(0)?;
            let offsets = offsets.read(&queue)?.to_slice()?.into_vec();

            let values = values.read(&queue)?.to_slice()?.into_vec();

            debug_assert_eq!(offsets.len(), values.len());

            Result::<_, TCError>::Ok(futures::stream::iter(
                offsets
                    .into_iter()
                    .zip(values)
                    .map(Result::<_, TCError>::Ok),
            ))
        })
        .try_flatten();

    Box::pin(offsets)
}

#[inline]
fn size_hint(size: u64) -> usize {
    size.try_into().ok().unwrap_or_else(|| usize::MAX)
}

#[inline]
fn unwrap_row<T>(mut row: Vec<Number>) -> (Coord, T)
where
    Number: CastInto<T> + CastInto<u64>,
{
    let n = row.pop().expect("n").cast_into();
    let coord = row.into_iter().map(|i| i.cast_into()).collect();
    (coord, n)
}

#[inline]
fn table_range(range: &Range) -> Result<b_table::Range<usize, Number>, TCError> {
    if range == &Range::default() {
        return Ok(b_table::Range::default());
    }

    let mut table_range = HashMap::new();

    for (x, bound) in range.iter().enumerate() {
        match bound {
            AxisRange::At(i) => {
                table_range.insert(x, b_table::ColumnRange::Eq(Number::from(*i)));
            }
            AxisRange::In(axis_range, 1) => {
                let start = Bound::Included(Number::from(axis_range.start));
                let stop = Bound::Excluded(Number::from(axis_range.end));
                table_range.insert(x, b_table::ColumnRange::In((start, stop)));
            }
            bound => {
                return Err(bad_request!(
                    "sparse tensor does not support axis bound {:?}",
                    bound
                ));
            }
        }
    }

    Ok(table_range.into())
}
