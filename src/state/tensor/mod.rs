mod base;
mod dense;
mod sparse;

pub enum Tensor {
    Dense(dense::BlockTensor),
    Sparse(sparse::SparseTensor),
}
