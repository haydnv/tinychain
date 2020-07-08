use std::marker::PhantomData;

mod base;
mod chunk;
mod dense;
mod index;
mod sparse;

pub enum Tensor<'a> {
    Dense(dense::BlockTensor<'a>),
    Sparse(sparse::SparseTensor),
    Phantom(PhantomData<&'a dense::BlockTensor<'a>>),
}
