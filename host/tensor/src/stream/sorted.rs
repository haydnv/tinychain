use std::fmt;

use afarray::{Array, ArrayExt, CoordUnique, Coords, Offsets};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use log::{debug, trace};
use safecast::AsType;

use tc_error::*;
use tc_table::{BTreeType, Node, NodeId};
use tc_transact::fs::{Dir, DirRead, DirWrite, File};
use tc_transact::{Transaction, TxnId};
use tc_value::{Number, UIntType};

use crate::dense::{BlockListFile, PER_BLOCK};
use crate::{Coord, DenseAccess, Ordinal, Shape, TensorAccess, TensorType};

use super::ReadValueAt;

pub async fn sorted_coords<FD, FS, D, T, C>(
    txn: &T,
    shape: Shape,
    coords: C,
) -> TCResult<impl Stream<Item = TCResult<Coords>> + Unpin>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Ordinal, Array>,
    FS: File<NodeId, Node>,
    <D::Read as DirRead>::FileEntry: AsType<FD> + AsType<FS>,
    <D::Write as DirWrite>::FileClass: From<BTreeType> + From<TensorType>,
    C: Stream<Item = TCResult<Coords>> + Unpin + Send,
{
    let txn_id = *txn.id();
    let file: FD = txn
        .context()
        .create_file_unique(txn_id, TensorType::Dense)
        .await?;

    let coords = coords.inspect_ok(|block| trace!("coord block len is {}", block.len()));
    let offsets = sort_coords::<FD, FS, D, T, _>(file, txn_id, coords, shape.clone()).await?;
    debug!("sorted coords");

    let offsets = offsets.block_stream(txn.clone()).await?;
    let offsets = offsets.map_ok(|array| array.type_cast());

    let coords = offsets_to_coords(shape.clone(), offsets);
    let coords = CoordUnique::new(coords, shape.to_vec(), PER_BLOCK);
    Ok(coords)
}

pub async fn sorted_values<'a, FD, FS, T, D, A, C>(
    txn: T,
    source: A,
    coords: C,
) -> TCResult<impl Stream<Item = TCResult<(Coord, Number)>>>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Ordinal, Array>,
    FS: File<NodeId, Node>,
    <D::Read as DirRead>::FileEntry: AsType<FD> + AsType<FS>,
    <D::Write as DirWrite>::FileClass: From<BTreeType> + From<TensorType>,
    A: TensorAccess + ReadValueAt<D, Txn = T> + Clone + fmt::Display + 'a,
    C: Stream<Item = TCResult<Coords>> + Send + Unpin + 'a,
{
    debug!("sort values by coordinate for {}", source);

    let coords = sorted_coords::<FD, FS, D, T, C>(&txn, source.shape().clone(), coords).await?;

    let buffered = coords
        .map_ok(|coords| stream::iter(coords.to_vec()).map(TCResult::Ok))
        .try_flatten()
        .inspect_ok(|coord| trace!("read value at {:?} to sort", coord))
        .map_ok(move |coord| source.clone().read_value_at(txn.clone(), coord))
        .try_buffered(num_cpus::get())
        .inspect_ok(|(coord, value)| trace!("value at {:?} is {}", coord, value));

    Ok(buffered)
}

async fn sort_coords<FD, FS, D, T, S>(
    file: FD,
    txn_id: TxnId,
    coords: S,
    shape: Shape,
) -> TCResult<BlockListFile<FD, FS, D, T>>
where
    FD: File<Ordinal, Array>,
    FS: File<NodeId, Node>,
    D: Dir,
    T: Transaction<D>,
    S: Stream<Item = TCResult<Coords>> + Send + Unpin,
{
    debug!("sort coords for Tensor with shape {}", shape);

    let blocks =
        coords_to_offsets(shape, coords).map_ok(|block: Offsets| ArrayExt::from(block).into());

    let block_list =
        BlockListFile::from_blocks(file, txn_id, None, UIntType::U64.into(), Box::pin(blocks))
            .await?;

    block_list.merge_sort(txn_id).await?;
    Ok(block_list)
}

fn coords_to_offsets<S: Stream<Item = TCResult<Coords>> + Unpin>(
    shape: Shape,
    coords: S,
) -> impl Stream<Item = TCResult<Offsets>> {
    debug!("coords to offsets with shape {}", shape);

    coords.map_ok(move |coords| {
        debug_assert_eq!(coords.ndim(), shape.len());
        coords.to_offsets(&shape)
    })
}

fn offsets_to_coords<'a, S: Stream<Item = TCResult<Offsets>> + Unpin + 'a>(
    shape: Shape,
    offsets: S,
) -> impl Stream<Item = TCResult<Coords>> + Unpin + 'a {
    offsets.map_ok(move |block| Coords::from_offsets(block, &shape))
}
