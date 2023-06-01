use std::fmt;

mod access;
mod base;
mod schema;
mod stream;

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, TryFutureExt};
use ha_ndarray::{CDatatype, NDArrayMath, NDArrayRead, NDArrayTransform};

use tc_error::*;
use tc_value::{DType, Number};

use super::{Axes, Coord, Range, TensorInstance};

const BLOCK_SIZE: usize = 4_096;

pub use base::SparseTensorTable;
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
