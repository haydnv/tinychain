mod array;
mod base;
mod bounds;
mod dense;
mod sparse;
mod stream;

pub type Array = array::Array;

pub enum Tensor {
    Dense(dense::BlockTensor),
    Sparse(sparse::SparseTensor),
}
