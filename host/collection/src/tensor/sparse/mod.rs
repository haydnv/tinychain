use std::fmt;
use std::marker::PhantomData;

mod access;
mod base;
mod schema;
mod stream;

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, TryFutureExt};
use ha_ndarray::{CDatatype, NDArrayMath, NDArrayRead, NDArrayTransform};
use safecast::{AsType, CastFrom};

use tc_error::*;
use tc_value::{DType, Number, NumberType};

use super::{Axes, Coord, Range, Shape, TensorInstance, TensorTransform};

use access::{
    SparseAccess, SparseBroadcast, SparseExpand, SparseReshape, SparseSlice, SparseTranspose,
};

const BLOCK_SIZE: usize = 4_096;

pub use base::SparseBase;
pub use schema::{IndexSchema, Schema};

pub type Blocks<C, V> = Pin<Box<dyn Stream<Item = Result<(C, V), TCError>> + Send>>;
pub type Elements<T> = Pin<Box<dyn Stream<Item = Result<(Coord, T), TCError>> + Send>>;
pub type Node = b_table::b_tree::Node<Vec<Vec<Number>>>;

#[async_trait]
pub trait SparseInstance: TensorInstance + fmt::Debug {
    type CoordBlock: NDArrayRead<DType = u64> + NDArrayMath + NDArrayTransform;
    type ValueBlock: NDArrayRead<DType = Self::DType>;
    type Blocks: Stream<Item = Result<(Self::CoordBlock, Self::ValueBlock), TCError>> + Send;
    type DType: CDatatype + DType;

    async fn blocks(self, range: Range, order: Axes) -> Result<Self::Blocks, TCError>;

    async fn elements(self, range: Range, order: Axes) -> Result<Elements<Self::DType>, TCError>;

    async fn filled_at(
        self,
        range: Range,
        axes: Axes,
    ) -> Result<stream::FilledAt<Elements<Self::DType>>, TCError>
    where
        Self: Sized,
    {
        let ndim = self.ndim();

        let elided = (0..ndim).filter(|x| !axes.contains(x));

        let mut order = Vec::with_capacity(ndim);
        order.copy_from_slice(&axes);
        order.extend(elided);

        self.elements(range, order)
            .map_ok(|elements| stream::FilledAt::new(elements, axes, ndim))
            .await
    }

    async fn read_value(&self, coord: Coord) -> Result<Self::DType, TCError>;
}

pub struct SparseTensor<FE, A> {
    accessor: A,
    phantom: PhantomData<FE>,
}

impl<FE, A> SparseTensor<FE, A> {
    pub fn into_inner(self) -> A {
        self.accessor
    }
}

impl<FE, A: Clone> Clone for SparseTensor<FE, A> {
    fn clone(&self) -> Self {
        Self {
            accessor: self.accessor.clone(),
            phantom: PhantomData,
        }
    }
}

impl<FE, A> TensorInstance for SparseTensor<FE, A>
where
    FE: Send + Sync + 'static,
    A: SparseInstance,
{
    fn dtype(&self) -> NumberType {
        self.accessor.dtype()
    }

    fn shape(&self) -> &Shape {
        self.accessor.shape()
    }
}

impl<FE, A> TensorTransform for SparseTensor<FE, A>
where
    FE: AsType<Node> + Send + Sync + 'static,
    A: SparseInstance + Into<SparseAccess<FE, A::DType>>,
    A::DType: CastFrom<Number> + fmt::Debug,
    Number: From<A::DType>,
{
    type Broadcast = SparseTensor<FE, SparseBroadcast<FE, A::DType>>;
    type Expand = SparseTensor<FE, SparseExpand<A>>;
    type Reshape = SparseTensor<FE, SparseReshape<A>>;
    type Slice = SparseTensor<FE, SparseSlice<A>>;
    type Transpose = SparseTensor<FE, SparseTranspose<A>>;

    fn broadcast(self, shape: Shape) -> TCResult<Self::Broadcast> {
        let accessor = SparseBroadcast::new(self.accessor, shape)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn expand(self, axes: Axes) -> TCResult<Self::Expand> {
        let accessor = SparseExpand::new(self.accessor, axes)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn reshape(self, shape: Shape) -> TCResult<Self::Reshape> {
        let accessor = SparseReshape::new(self.accessor, shape)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn slice(self, range: Range) -> TCResult<Self::Slice> {
        let accessor = SparseSlice::new(self.accessor, range)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }

    fn transpose(self, permutation: Option<Axes>) -> TCResult<Self::Transpose> {
        let accessor = SparseTranspose::new(self.accessor, permutation)?;

        Ok(SparseTensor {
            accessor,
            phantom: PhantomData,
        })
    }
}

impl<FE, A> From<A> for SparseTensor<FE, SparseAccess<FE, A::DType>>
where
    A: SparseInstance + Into<SparseAccess<FE, A::DType>>,
{
    fn from(accessor: A) -> Self {
        Self {
            accessor: accessor.into(),
            phantom: PhantomData,
        }
    }
}
