mod base;
mod bounds;
mod chunk;
mod dense;
mod sparse;
mod stream;

pub enum Tensor {
    Dense(dense::BlockTensor),
    Sparse(sparse::SparseTensor),
}
