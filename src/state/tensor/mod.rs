mod base;
mod chunk;
mod dense;
mod index;
mod sparse;
mod stream;

pub enum Tensor {
    Dense(dense::BlockTensor),
    Sparse(sparse::SparseTensor),
}
