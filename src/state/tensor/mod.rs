mod base;
mod chunk;
mod dense;
mod index;
mod sparse;

pub enum Tensor {
    Dense(dense::BlockTensor),
    Sparse(sparse::SparseTensor),
}
