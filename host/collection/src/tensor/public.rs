use std::convert::TryInto;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

use futures::future::{self, Future, TryFutureExt};
use futures::stream::{self, FuturesUnordered, StreamExt, TryStreamExt};
use log::debug;
use safecast::*;

use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::fs;
use tc_transact::public::helpers::{AttributeHandler, SelfHandlerOwned};
use tc_transact::public::{GetHandler, Handler, PostHandler, PutHandler, Route, StateInstance};
use tc_transact::{Transaction, TxnId};
use tc_value::{FloatType, Number, NumberClass, NumberInstance, NumberType, Value, ValueType};
use tcgeneric::{label, Id, Label, PathSegment, TCBoxTryFuture, ThreadSafe, Tuple};

use super::{
    broadcast, broadcast_shape, Axes, AxisRange, Dense, DenseBase, DenseCacheFile, DenseView, Node,
    Range, Schema, Shape, Sparse, SparseBase, SparseView, Tensor, TensorBoolean,
    TensorBooleanConst, TensorCast, TensorCompare, TensorCompareConst, TensorConvert,
    TensorDiagonal, TensorInstance, TensorMath, TensorMathConst, TensorRead, TensorReduce,
    TensorTransform, TensorTrig, TensorType, TensorUnary, TensorUnaryBoolean, TensorWrite,
    TensorWriteDual,
};

const AXIS: Label = label("axis");
const KEEPDIMS: Label = label("keepdims");
const RIGHT: Label = label("r");
const TENSOR: Label = label("tensor");
const TENSORS: Label = label("tensors");

const MEAN: f64 = 0.0;
const STD: f64 = 0.0;

struct BroadcastHandler<T> {
    tensor: T,
}

impl<'a, State, T> Handler<'a, State> for BroadcastHandler<T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorInstance + TensorTransform + Send + Sync + 'a,
    Tensor<State::Txn, State::FE>: From<T::Broadcast>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let shape = Shape::try_cast_from(key, |v| {
                    TCError::unexpected(v, "a Tensor shape for broadcasting")
                })?;

                let shape = broadcast_shape(self.tensor.shape(), &shape)?;

                self.tensor
                    .broadcast(shape)
                    .map(Tensor::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for BroadcastHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct CastHandler<T> {
    tensor: T,
}

impl<'a, State, T> Handler<'a, State> for CastHandler<T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorCast + Send + Sync + 'a,
    Tensor<State::Txn, State::FE>: From<T::Cast>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let dtype =
                    ValueType::try_cast_from(key, |v| TCError::unexpected(v, "a Number class"))?;

                let dtype = dtype.try_into()?;
                TensorCast::cast_into(self.tensor, dtype)
                    .map(Tensor::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for CastHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct ConcatenateHandler;

impl ConcatenateHandler {
    async fn concatenate_axis<State>(
        txn: &State::Txn,
        axis: usize,
        dtype: NumberType,
        tensors: Vec<Tensor<State::Txn, State::FE>>,
    ) -> TCResult<Tensor<State::Txn, State::FE>>
    where
        State: StateInstance,
        State::FE: DenseCacheFile + AsType<Node> + Clone,
        Tensor<State::Txn, State::FE>: TensorConvert<Dense = Dense<State::Txn, State::FE>>,
    {
        const ERR_OFF_AXIS: &str = "Tensors to concatenate must have the same off-axis dimensions";

        let ndim = tensors[0].ndim();
        let mut offsets = Vec::with_capacity(tensors.len());
        let mut shape_out = tensors[0].shape().to_vec();
        shape_out[axis] = 0;

        for tensor in &tensors {
            tensor.shape().validate()?;

            if tensor.ndim() != ndim {
                return Err(bad_request!(
                    "Tensors to concatenate must have the same dimensions"
                ));
            }

            let shape = tensor.shape();

            for x in 0..axis {
                let dim = shape[x];
                if dim != shape_out[x] {
                    return Err(bad_request!("{}", ERR_OFF_AXIS));
                }
            }

            if axis < tensor.ndim() {
                for x in (axis + 1)..ndim {
                    if shape[x] != shape_out[x] {
                        return Err(bad_request!("{}", ERR_OFF_AXIS));
                    }
                }
            }

            shape_out[axis] += shape[axis];
            if let Some(offset) = offsets.last().cloned() {
                offsets.push(offset + shape[axis]);
            } else {
                offsets.push(shape[axis]);
            }
        }

        let range: Range = shape_out.iter().map(|dim| AxisRange::all(*dim)).collect();
        let concatenated = constant::<State>(txn, shape_out.into(), dtype.zero()).await?;
        debug!("concantenation shape is {:?}", concatenated.shape());

        let txn_id = *txn.id();
        let mut writes: FuturesUnordered<_> = tensors
            .into_iter()
            .zip(offsets)
            .map(|(tensor, offset)| {
                debug!("concatenate {:?} along axis {}", tensor, axis);
                let mut range = range.clone();
                range[axis] = AxisRange::In((offset - tensor.shape()[axis])..offset, 1);
                concatenated
                    .clone()
                    .write(txn_id, range, tensor.into_dense())
            })
            .collect();

        while let Some(()) = writes.try_next().await? {
            // no-op
        }

        Ok(Tensor::Dense(concatenated))
    }
}

impl<'a, State> Handler<'a, State> for ConcatenateHandler
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TensorConvert<Dense = Dense<State::Txn, State::FE>>,
    Vec<Tensor<State::Txn, State::FE>>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let tensors: Vec<Tensor<_, _>> = params.require(&TENSORS.into())?;
                let axis: Value = params.or_default(&AXIS.into())?;
                params.expect_empty()?;

                if tensors.is_empty() {
                    return Err(bad_request!("no Tensors to concatenate"));
                }

                let dtype = tensors
                    .iter()
                    .map(TensorInstance::dtype)
                    .fold(tensors[0].dtype(), Ord::max);

                let axis = if axis.is_some() {
                    cast_axis(axis, tensors[0].ndim())?
                } else {
                    0
                };

                if axis > tensors[0].ndim() {
                    return Err(bad_request!(
                        "axis {} is out of bounds for {:?}",
                        axis,
                        tensors[0]
                    ));
                }

                let tensor = Self::concatenate_axis::<State>(txn, axis, dtype, tensors).await?;
                Ok(State::from(tensor))
            })
        }))
    }
}

struct ConstantHandler;

impl<'a, State> Handler<'a, State> for ConstantHandler
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + Clone,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let (shape, value): (Vec<u64>, Number) =
                    key.try_cast_into(|v| TCError::unexpected(v, "a Tensor schema"))?;

                let shape = Shape::from(shape);
                constant::<State>(&txn, shape, value)
                    .map_ok(Tensor::from)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct CopyFromHandler;

impl<'a, State> Handler<'a, State> for CopyFromHandler
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            use fs::CopyFrom;

            Box::pin(async move {
                let source: Tensor<_, _> = params.require(&TENSOR.into())?;
                params.expect_empty()?;

                source.shape().validate()?;

                let copy = match source {
                    Tensor::Dense(source) => {
                        let dir = create_dir(txn).await?;
                        let copy = DenseBase::copy_from(txn, dir, source.into_view()).await?;

                        Tensor::Dense(Dense::Base(copy))
                    }
                    Tensor::Sparse(source) => {
                        let dir = create_dir(txn).await?;
                        let copy = SparseBase::copy_from(txn, dir, source.into_view()).await?;

                        Tensor::Sparse(Sparse::Base(copy))
                    }
                };

                Ok(State::from(copy))
            })
        }))
    }
}

struct CopyDenseHandler;

impl<'a, State> Handler<'a, State> for CopyDenseHandler
where
    State: StateInstance,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let schema =
                    Schema::try_cast_from(schema, |v| TCError::unexpected(v, "a Tensor schema"))?;

                schema.shape.validate()?;

                Err(not_implemented!(
                    "copy a dense tensor from a stream of values"
                ))
            })
        }))
    }
}

struct CopySparseHandler;

impl<'a, State> Handler<'a, State> for CopySparseHandler
where
    State: StateInstance,
    Value: TryCastFrom<State>,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let schema: Value = params.require(&label("schema").into())?;
                let schema: Schema =
                    schema.try_cast_into(|v| TCError::unexpected(v, "invalid Tensor schema"))?;

                schema.shape.validate()?;

                // let tensor = create_sparse(txn, schema).await?;
                //
                // let elements = elements
                //     .map(|r| {
                //         r.and_then(|state| {
                //             Value::try_cast_from(state, |s| {
                //                 TCError::unexpected(s, "a sparse Tensor element")
                //             })
                //         })
                //     })
                //     .map(|r| {
                //         r.and_then(|row| {
                //             row.try_cast_into(|v| {
                //                 bad_request!(
                //                     "sparse Tensor expected a (Coord, Number) tuple, found {}",
                //                     v,
                //                 )
                //             })
                //         })
                //     });
                //
                // elements
                //     .map_ok(|(coord, value)| tensor.write_value_at(*txn.id(), coord, value))
                //     .try_buffer_unordered(num_cpus::get())
                //     .try_fold((), |(), ()| future::ready(Ok(())))
                //     .await?;
                //
                // Ok(Collection::Tensor(tensor.into()).into())
                Err(bad_request!("copy a collection into a sparse tensor"))
            })
        }))
    }
}

struct CreateHandler {
    class: TensorType,
}

impl<'a, State> Handler<'a, State> for CreateHandler
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let _schema: Schema =
                    key.try_cast_into(|v| TCError::unexpected(v, "a Tensor schema"))?;

                // create_tensor(self.class, schema, txn)
                //     .map_ok(State::from)
                //     .await

                Err(not_implemented!("create tensor"))
            })
        }))
    }
}

struct LoadHandler {
    class: Option<TensorType>,
}

impl<'a, State> Handler<'a, State> for LoadHandler
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let (schema, elements): (Value, Value) =
                    key.try_cast_into(|v| TCError::unexpected(v, "a Tensor schema"))?;

                let txn_id = *txn.id();

                if elements.matches::<Vec<(Vec<u64>, Number)>>() {
                    let elements: Vec<(Vec<u64>, Number)> = elements
                        .opt_cast_into()
                        .expect("tensor coordinate elements");

                    let schema = if schema.matches::<Scalar>() {
                        schema.opt_cast_into().expect("load tensor schema")
                    } else if schema.matches::<Shape>() {
                        let shape = schema.opt_cast_into().expect("load tensor shape");
                        let dtype = if elements.is_empty() {
                            NumberType::Float(FloatType::F32)
                        } else {
                            elements[0].1.class()
                        };

                        Schema { shape, dtype }
                    } else {
                        return Err(TCError::unexpected(schema, "a Tensor schema"));
                    };

                    let class = self.class.unwrap_or(TensorType::Sparse);
                    // let tensor = create_tensor(class, schema, txn).await?;
                    //
                    // stream::iter(elements)
                    //     .map(|(coord, value)| tensor.write_value_at(txn_id, coord, value))
                    //     .buffer_unordered(num_cpus::get())
                    //     .try_fold((), |(), ()| future::ready(Ok(())))
                    //     .await?;
                    //
                    // Ok(State::from(tensor))
                    Err(not_implemented!("load sparse tensor"))
                } else if elements.matches::<Vec<Number>>() {
                    let elements: Vec<Number> = elements.opt_cast_into().expect("tensor elements");
                    if elements.is_empty() {
                        return Err(bad_request!(
                            "a dense Tensor cannot be loaded from a zero-element Tuple",
                        ));
                    }

                    let schema = if schema.matches::<Schema>() {
                        schema.opt_cast_into().expect("load tensor schema")
                    } else if schema.matches::<Shape>() {
                        let shape = schema.opt_cast_into().expect("load tensor shape");
                        Schema {
                            shape,
                            dtype: elements[0].class(),
                        }
                    } else {
                        return Err(TCError::unexpected(schema, "a Tensor schema"));
                    };

                    if elements.len() as u64 != schema.shape.size() {
                        return Err(bad_request!(
                            "wrong number of elements for Tensor with shape {:?}: {:?}",
                            schema.shape,
                            elements
                        ));
                    }

                    if self.class == Some(TensorType::Sparse) {
                        return Err(bad_request!(
                            "loading all elements {:?} of a Sparse tensor does not make sense",
                            elements,
                        ));
                    }

                    // let txn_id = *txn.id();
                    // let file = create_file(txn).await?;
                    // let elements = stream::iter(elements).map(Ok);
                    // DenseTensorFile::from_values(file, txn_id, schema.shape, schema.dtype, elements)
                    //     .map_ok(Dense::from)
                    //     .map_ok(Tensor::from)
                    //     .map_ok(State::from)
                    //     .await
                    Err(not_implemented!("load dense tensor"))
                } else {
                    Err(bad_request!("tensor elements must be a Tuple of Numbers or a Tuple of (Coord, Number) pairs, not {}", elements))
                }
            })
        }))
    }
}

struct EyeHandler;

impl<'a, State> Handler<'a, State> for EyeHandler
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TensorWrite,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let size = u64::try_cast_from(key, |v| {
                    TCError::unexpected(v, "the size of an identity tensor")
                })?;

                let schema = Schema::from((NumberType::Bool, vec![size, size].into()));
                let tensor = create_sparse::<State>(txn, schema).await?;

                stream::iter(0..size)
                    .map(|i| (vec![i, i], true.into()))
                    .map(|(coord, value)| tensor.write_value_at(*txn.id(), coord, value))
                    .buffer_unordered(num_cpus::get())
                    .try_fold((), |(), ()| future::ready(Ok(())))
                    .await?;

                Ok(State::from(Tensor::Sparse(tensor)))
            })
        }))
    }
}

struct DiagonalHandler<T> {
    tensor: T,
}

impl<'a, State, T> Handler<'a, State> for DiagonalHandler<T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorDiagonal + Send + 'a,
    Tensor<State::Txn, State::FE>: From<T::Diagonal>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                key.expect_none()?;

                self.tensor.diagonal().map(Tensor::from).map(State::from)
            })
        }))
    }
}

impl<T> From<T> for DiagonalHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct ExpandHandler<T> {
    tensor: T,
}

impl<'a, State, T> Handler<'a, State> for ExpandHandler<T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorInstance + TensorTransform + Send + 'a,
    Tensor<State::Txn, State::FE>: From<T::Expand>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                self.tensor.shape().validate()?;

                let axes = if key.is_none() {
                    vec![self.tensor.ndim()]
                } else {
                    cast_axes(key, self.tensor.ndim())?
                };

                self.tensor.expand(axes).map(Tensor::from).map(State::from)
            })
        }))
    }
}

impl<T> From<T> for ExpandHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

// struct RandomNormalHandler;
//
// impl<'a, State> Handler<'a, State> for RandomNormalHandler where State: StateInstance {
//     fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
//     where
//         'b: 'a,
//     {
//         Some(Box::new(|txn, key| {
//             Box::pin(async move {
//                 let shape = key.try_cast_into(|v| TCError::unexpected(v, "a Tensor shape"))?;
//
//                 let file = create_file(&txn).await?;
//
//                 let tensor = DenseBase::random_normal(
//                     file,
//                     *txn.id(),
//                     shape,
//                     FloatType::F64,
//                     MEAN.into(),
//                     STD.into(),
//                 )
//                 .map_ok(Dense::Base)
//                 .map_ok(Tensor::Dense)
//                 .await?;
//
//                 Ok(State::from(tensor))
//             })
//         }))
//     }
//
//     fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
//     where
//         'b: 'a,
//     {
//         Some(Box::new(|txn, mut params| {
//             Box::pin(async move {
//                 let shape: Value = params.require(&label("shape").into())?;
//                 let shape: Shape =
//                     shape.try_cast_into(|v| TCError::unexpected(v, "a Tensor shape"))?;
//
//                 let mean = params.option(&label("mean").into(), || MEAN.into())?;
//                 let std = params.option(&label("std").into(), || STD.into())?;
//                 params.expect_empty()?;
//
//                 let file = create_file(&txn).await?;
//
//                 let tensor = DenseBase::random_normal(
//                     file,
//                     *txn.id(),
//                     shape.into(),
//                     FloatType::F64,
//                     mean,
//                     std,
//                 )
//                 .map_ok(Dense::Base)
//                 .map_ok(Tensor::Dense)
//                 .await?;
//
//                 Ok(State::from(tensor))
//             })
//         }))
//     }
// }
//
// struct RandomUniformHandler;
//
// impl<'a, State> Handler<'a, State> for RandomUniformHandler where State: StateInstance {
//     fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
//     where
//         'b: 'a,
//     {
//         Some(Box::new(|txn, key| {
//             Box::pin(async move {
//                 let shape = key.try_cast_into(|v| TCError::unexpected(v, "a Tensor shape"))?;
//
//                 let file = create_file(&txn).await?;
//
//                 let tensor = DenseBase::random_uniform(file, *txn.id(), shape, FloatType::F64)
//                     .map(Dense::Base)
//                     .map(Tensor::from)?;
//
//                 Ok(State::from(tensor))
//             })
//         }))
//     }
// }

struct RangeHandler;

impl<'a, State> Handler<'a, State> for RangeHandler
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.matches::<(Vec<u64>, Number, Number)>() {
                    // let (shape, start, stop): (Vec<u64>, Number, Number) =
                    //     key.opt_cast_into().unwrap();
                    //
                    // let dir = create_dir(&txn).await?;

                    // DenseBase::range(dir, *txn.id(), shape, start, stop)
                    //     .map_ok(Dense::Base)
                    //     .map_ok(Tensor::Dense)
                    //     .map_ok(State::from)
                    //     .await
                    Err(not_implemented!("dense tensor range constructor"))
                } else {
                    Err(TCError::unexpected(key, "a Tensor schema"))
                }
            })
        }))
    }
}

struct ReshapeHandler<T> {
    tensor: T,
}

impl<'a, State, T> Handler<'a, State> for ReshapeHandler<T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorInstance + TensorTransform + Send + 'a,
    Tensor<State::Txn, State::FE>: From<T::Reshape>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let shape = key.try_into()?;
                let shape = cast_shape(self.tensor.shape(), shape)?;
                self.tensor
                    .reshape(shape.into())
                    .map(Tensor::from)
                    .map(State::from)
            })
        }))
    }
}

impl<T> From<T> for ReshapeHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct TransposeHandler<T> {
    tensor: T,
}

impl<'a, State, T> Handler<'a, State> for TransposeHandler<T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorTransform + Send + 'a,
    Tensor<State::Txn, State::FE>: From<T::Transpose>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                let transpose = if key.is_none() {
                    self.tensor.transpose(None)
                } else {
                    let permutation =
                        key.try_cast_into(|v| TCError::unexpected(v, "a Tensor permutation"))?;

                    self.tensor.transpose(Some(permutation))
                };

                transpose.map(Tensor::from).map(State::from)
            })
        }))
    }
}

impl<T> From<T> for TransposeHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

impl<State> Route<State> for TensorType
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Vec<Tensor<State::Txn, State::FE>>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            return Some(Box::new(CreateHandler { class: *self }));
        }

        if path.len() == 1 {
            match self {
                Self::Dense => match path[0].as_str() {
                    "copy_from" => Some(Box::new(CopyDenseHandler)),
                    "concatenate" => Some(Box::new(ConcatenateHandler)),
                    "constant" => Some(Box::new(ConstantHandler)),
                    "load" => Some(Box::new(LoadHandler { class: Some(*self) })),
                    "range" => Some(Box::new(RangeHandler)),
                    // "random" if path.len() == 1 => Some(Box::new(RandomUniformHandler)),
                    _ => None,
                },
                Self::Sparse => match path[0].as_str() {
                    "copy_from" => Some(Box::new(CopySparseHandler)),
                    "eye" => Some(Box::new(EyeHandler)),
                    "load" => Some(Box::new(LoadHandler { class: Some(*self) })),
                    _ => None,
                },
            }
        } else if path.len() == 2 {
            match self {
                Self::Dense => match path[0].as_str() {
                    "random" => match path[1].as_str() {
                        // "normal" => Some(Box::new(RandomNormalHandler)),
                        // "uniform" => Some(Box::new(RandomUniformHandler)),
                        _ => None,
                    },
                    _ => None,
                },
                Self::Sparse => None,
            }
        } else {
            None
        }
    }
}

struct DualHandler<Txn, FE> {
    tensor: Tensor<Txn, FE>,
    op: fn(Tensor<Txn, FE>, Tensor<Txn, FE>) -> TCResult<Tensor<Txn, FE>>,
    op_const: fn(Tensor<Txn, FE>, Number) -> TCResult<Tensor<Txn, FE>>,
    op_name: &'static str,
}

impl<Txn, FE> DualHandler<Txn, FE> {
    fn new<T>(
        tensor: T,
        op: fn(Tensor<Txn, FE>, Tensor<Txn, FE>) -> TCResult<Tensor<Txn, FE>>,
        op_const: fn(Tensor<Txn, FE>, Number) -> TCResult<Tensor<Txn, FE>>,
        op_name: &'static str,
    ) -> Self
    where
        Tensor<Txn, FE>: From<T>,
    {
        Self {
            tensor: tensor.into(),
            op,
            op_const,
            op_name,
        }
    }
}

impl<'a, State> Handler<'a, State> for DualHandler<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, r| {
            Box::pin(async move {
                let r = r.try_into()?;

                self.tensor.shape().validate()?;

                (self.op_const)(self.tensor, r).map(State::from)
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let l = self.tensor;
                let r = params.remove::<Id>(&RIGHT.into()).ok_or_else(|| {
                    TCError::unexpected(&params, "missing right-hand-side parameter r")
                })?;

                params.expect_empty()?;

                l.shape().validate()?;

                if r.matches::<Tensor<_, _>>() {
                    let r = Tensor::<_, _>::opt_cast_from(r).expect("tensor");
                    r.shape().validate()?;

                    if l.shape() == r.shape() {
                        (self.op)(l, r).map(State::from)
                    } else {
                        let (l, r) = broadcast(l, r)?;
                        (self.op)(l, r).map(State::from)
                    }
                } else if r.matches::<Number>() {
                    let r = r.opt_cast_into().expect("number");
                    (self.op_const)(l, r).map(State::from)
                } else {
                    Err(bad_request!("expected a Tensor or Number, not {r:?}"))
                }
            })
        }))
    }
}

// TODO: should this be more general, like `DualHandlerWithDefaultArgument`?
struct LogHandler<Txn, FE> {
    tensor: Tensor<Txn, FE>,
}

impl<Txn, FE> LogHandler<Txn, FE> {
    fn new<T>(tensor: T) -> Self
    where
        Tensor<Txn, FE>: From<T>,
    {
        Self {
            tensor: tensor.into(),
        }
    }
}

impl<'a, State> Handler<'a, State> for LogHandler<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Number: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, r| {
            Box::pin(async move {
                self.tensor.shape().validate()?;

                let log = if r.is_none() {
                    self.tensor.ln()?
                } else {
                    let base = r.try_into()?;
                    self.tensor.log_const(base)?
                };

                Ok(State::from(log))
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let r: State = params.or_default(&RIGHT.into())?;
                params.expect_empty()?;

                let l = self.tensor;
                l.shape().validate()?;

                let log = if r.matches::<Tensor<_, _>>() {
                    let base = Tensor::<_, _>::opt_cast_from(r).expect("tensor");
                    base.shape().validate()?;

                    if l.shape() == base.shape() {
                        l.log(base)
                    } else {
                        let (l, base) = broadcast(l, base)?;
                        l.log(base)
                    }
                } else if r.matches::<Number>() {
                    let base = Number::opt_cast_from(r).expect("numeric constant");
                    l.log_const(base)
                } else {
                    Err(bad_request!("a Tensor or Number"))
                }?;

                Ok(State::from(log))
            })
        }))
    }
}

// struct MatMulHandler<Txn, FE> {
//     tensor: Tensor<Txn, FE>,
// }
//
// impl<Txn, FE> MatMulHandler<Txn, FE> {
//     fn new<T: Into<Tensor<Txn, FE>>>(tensor: T) -> Self {
//         Self {
//             tensor: tensor.into(),
//         }
//     }
// }
//
// impl<'a, State> Handler<'a, State> for MatMulHandler<State::Txn, State::FE> where State: StateInstance {
//     fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
//     where
//         'b: 'a,
//     {
//         Some(Box::new(|_txn, mut params| {
//             Box::pin(async move {
//                 let right: Tensor<_, _> = params.require(&RIGHT.into())?;
//                 params.expect_empty()?;
//
//                 self.tensor.matmul(right).map_ok(Tensor::from).await
//             })
//         }))
//     }
// }

struct NormHandler<State: StateInstance> {
    tensor: Tensor<State::Txn, State::FE>,
}

impl<State> NormHandler<State>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
{
    async fn call(
        tensor: Tensor<State::Txn, State::FE>,
        txn: State::Txn,
        axis: Option<usize>,
        keepdims: bool,
    ) -> TCResult<State> {
        if let Some(axis) = axis {
            debug!("norm of {tensor:?} at axis {axis}");

            return tensor
                .pow_const(2i32.into())
                .and_then(|pow| pow.sum(vec![axis], keepdims))
                .and_then(|sum| sum.pow_const(0.5f32.into()))
                .map(State::from);
        } else if tensor.ndim() <= 2 {
            if keepdims {
                Err(not_implemented!("matrix norm with keepdims"))
            } else {
                let squared = tensor.pow_const(2i32.into())?;
                let summed = squared.sum_all(*txn.id()).await?;
                Ok(Value::from(summed.pow(0.5f32.into())).into())
            }
        } else {
            debug!("norm of {tensor:?}, keepdims is {keepdims}");

            tensor
                .pow_const(2i32.into())
                .and_then(|pow| {
                    let axes = vec![pow.ndim() - 1, pow.ndim() - 2];
                    pow.sum(axes, keepdims)
                })
                .and_then(|sum| sum.pow_const(0.5f32.into()))
                .map(State::from)
        }
    }
}

impl<'a, State> Handler<'a, State> for NormHandler<State>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let axis = if key.is_some() {
                    cast_axis(key, self.tensor.ndim()).map(Some)?
                } else {
                    None
                };

                Self::call(self.tensor, txn.clone(), axis, false).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let axis = if params.contains_key::<Id>(&AXIS.into()) {
                    let axis = params.require(&AXIS.into())?;
                    cast_axis(axis, self.tensor.ndim()).map(Some)?
                } else {
                    None
                };

                let keepdims = params.or_default(&KEEPDIMS.into())?;

                params.expect_empty()?;

                Self::call(self.tensor, txn.clone(), axis, keepdims).await
            })
        }))
    }
}

impl<State: StateInstance> From<Tensor<State::Txn, State::FE>> for NormHandler<State> {
    fn from(tensor: Tensor<State::Txn, State::FE>) -> Self {
        Self { tensor }
    }
}

struct ReduceHandler<State, T: TensorReduce> {
    tensor: T,
    reduce: fn(T, Axes, bool) -> TCResult<<T as TensorReduce>::Reduce>,
    reduce_all: fn(T, TxnId) -> TCBoxTryFuture<'static, Number>,
    state: PhantomData<State>,
}

impl<State, T: TensorReduce> ReduceHandler<State, T> {
    fn new(
        tensor: T,
        reduce: fn(T, Axes, bool) -> TCResult<<T as TensorReduce>::Reduce>,
        reduce_all: fn(T, TxnId) -> TCBoxTryFuture<'static, Number>,
    ) -> Self {
        Self {
            tensor,
            reduce,
            reduce_all,
            state: PhantomData,
        }
    }
}

impl<State, T> ReduceHandler<State, T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorInstance + TensorReduce + Clone + Sync,
    Tensor<State::Txn, State::FE>: From<<T as TensorReduce>::Reduce>,
{
    async fn call(self, txn_id: TxnId, axes: Option<Axes>, keepdims: bool) -> TCResult<State> {
        if axes.is_none() && keepdims {
            Err(not_implemented!("reduce all axes but keep dimensions"))
        } else if axes.is_none() || axes.as_ref().map(|x| x.len()) == Some(self.tensor.ndim()) {
            (self.reduce_all)(self.tensor, txn_id)
                .map_ok(Value::from)
                .map_ok(State::from)
                .await
        } else {
            (self.reduce)(self.tensor, axes.expect("axes"), keepdims)
                .map(Tensor::from)
                .map(State::from)
        }
    }
}

impl<'a, State, T> Handler<'a, State> for ReduceHandler<State, T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorInstance + TensorReduce + Clone + Sync,
    Tensor<State::Txn, State::FE>: From<<T as TensorReduce>::Reduce>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let axis = if key.is_none() {
                    None
                } else {
                    cast_axes(key, self.tensor.ndim()).map(Some)?
                };

                self.call(*txn.id(), axis, false).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let axis = if params.contains_key::<Id>(&AXIS.into()) {
                    let axes = params.require(&AXIS.into())?;
                    cast_axes(axes, self.tensor.ndim()).map(Some)?
                } else {
                    None
                };

                let keepdims = params.or_default(&KEEPDIMS.into())?;
                params.expect_empty()?;

                self.call(*txn.id(), axis, keepdims).await
            })
        }))
    }
}

struct TensorHandler<T> {
    tensor: T,
}

impl<'a, State, T: 'a> Handler<'a, State> for TensorHandler<T>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    T: TensorInstance
        + TensorRead
        + TensorWrite
        + TensorWriteDual<Tensor<State::Txn, State::FE>>
        + TensorTransform
        + Clone
        + Send
        + Sync,
    Tensor<State::Txn, State::FE>: From<<T as TensorTransform>::Slice> + TryCastFrom<State>,
    Number: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!("GET Tensor: {}", key);
                let range = cast_range(self.tensor.shape(), key)?;

                if range.size() == 0 {
                    return Err(bad_request!(
                        "invalid bounds for tensor with shape {:?}: {:?}",
                        self.tensor.shape(),
                        range
                    ));
                }

                let shape = self.tensor.shape();
                if range.is_coord(shape) {
                    let coord = range.as_coord(shape).expect("tensor coordinate");

                    self.tensor
                        .read_value(*txn.id(), coord)
                        .map_ok(Value::from)
                        .map_ok(State::from)
                        .await
                } else {
                    self.tensor.slice(range).map(Tensor::from).map(State::from)
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            debug!("PUT Tensor: {} <- {:?}", key, value);
            Box::pin(write::<State, T>(self.tensor, *txn.id(), key, value))
        }))
    }
}

impl<T> From<T> for TensorHandler<T> {
    fn from(tensor: T) -> Self {
        Self { tensor }
    }
}

struct UnaryHandler<Txn, FE> {
    tensor: Tensor<Txn, FE>,
    op: fn(Tensor<Txn, FE>) -> TCResult<Tensor<Txn, FE>>,
    op_name: &'static str,
}

impl<Txn, FE> UnaryHandler<Txn, FE> {
    fn new(
        tensor: Tensor<Txn, FE>,
        op: fn(Tensor<Txn, FE>) -> TCResult<Tensor<Txn, FE>>,
        op_name: &'static str,
    ) -> Self {
        Self {
            tensor,
            op,
            op_name,
        }
    }
}

impl<'a, State> Handler<'a, State> for UnaryHandler<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    Tensor<State::Txn, State::FE>: TensorTransform<Slice = Tensor<State::Txn, State::FE>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                self.tensor.shape().validate()?;

                let tensor = if key.is_none() {
                    self.tensor
                } else {
                    let bounds = cast_range(self.tensor.shape(), key.into())?;
                    self.tensor.slice(bounds)?
                };

                (self.op)(tensor).map(State::from)
            })
        }))
    }
}

struct UnaryHandlerAsync<Txn, FE, F> {
    tensor: Tensor<Txn, FE>,
    op: fn(Tensor<Txn, FE>, TxnId) -> F,
    op_name: &'static str,
}

impl<'a, Txn, FE, F> UnaryHandlerAsync<Txn, FE, F> {
    fn new(
        tensor: Tensor<Txn, FE>,
        op: fn(Tensor<Txn, FE>, TxnId) -> F,
        op_name: &'static str,
    ) -> Self {
        Self {
            tensor,
            op,
            op_name,
        }
    }
}

impl<'a, State, F> Handler<'a, State> for UnaryHandlerAsync<State::Txn, State::FE, F>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    F: Future<Output = TCResult<bool>> + Send + 'a,
    Tensor<State::Txn, State::FE>: TensorTransform<Slice = Tensor<State::Txn, State::FE>>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                self.tensor.shape().validate()?;

                if key.is_none() {
                    (self.op)(self.tensor, *txn.id()).map_ok(State::from).await
                } else {
                    let bounds = cast_range(self.tensor.shape(), key.into())?;
                    let slice = self.tensor.slice(bounds)?;
                    (self.op)(slice, *txn.id()).map_ok(State::from).await
                }
            })
        }))
    }
}

impl<State> Route<State> for Dense<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Number: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self.clone().into(), path)
    }
}

impl<State> Route<State> for DenseBase<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Number: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(Dense::Base(self.clone()).into(), path)
    }
}

impl<State> Route<State> for DenseView<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Number: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self.clone().into(), path)
    }
}

impl<State> Route<State> for Sparse<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Number: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self.clone().into(), path)
    }
}

impl<State> Route<State> for SparseBase<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(Sparse::Base(self.clone()).into(), path)
    }
}

impl<State> Route<State> for SparseView<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self.clone().into(), path)
    }
}

impl<State> Route<State> for Tensor<State::Txn, State::FE>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        route(self.clone(), path)
    }
}

fn route<'a, State>(
    tensor: Tensor<State::Txn, State::FE>,
    path: &'a [PathSegment],
) -> Option<Box<dyn Handler<'a, State> + 'a>>
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>> + From<Tuple<Value>>,
    State::Class: From<NumberType>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Number: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    if path.is_empty() {
        Some(Box::new(TensorHandler::from(tensor.clone())))
    } else if path.len() == 1 {
        match path[0].as_str() {
            // attributes
            "dtype" => {
                return Some(Box::new(AttributeHandler::from(State::from(
                    State::Class::from(tensor.dtype()),
                ))));
            }

            "ndim" => {
                return Some(Box::new(AttributeHandler::from(Value::Number(
                    (tensor.ndim() as u64).into(),
                ))))
            }

            "shape" => {
                return Some(Box::new(AttributeHandler::from(
                    tensor
                        .shape()
                        .iter()
                        .map(|dim| Number::from(*dim))
                        .collect::<Tuple<Value>>(),
                )))
            }

            "size" => {
                return Some(Box::new(AttributeHandler::from(Value::Number(
                    tensor.size().into(),
                ))))
            }

            // basic math
            "add" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::add,
                TensorMathConst::add_const,
                "add",
            ))),
            "div" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::div,
                TensorMathConst::div_const,
                "div",
            ))),
            "log" => Some(Box::new(LogHandler::new(tensor))),
            "mul" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::mul,
                TensorMathConst::mul_const,
                "mul",
            ))),
            "pow" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::pow,
                TensorMathConst::pow_const,
                "pow",
            ))),
            "sub" => Some(Box::new(DualHandler::new(
                tensor,
                TensorMath::sub,
                TensorMathConst::sub_const,
                "sub",
            ))),

            // boolean ops
            "and" => Some(Box::new(DualHandler::new(
                tensor,
                TensorBoolean::and,
                TensorBooleanConst::and_const,
                "and",
            ))),
            "or" => Some(Box::new(DualHandler::new(
                tensor,
                TensorBoolean::or,
                TensorBooleanConst::or_const,
                "or",
            ))),
            "xor" => Some(Box::new(DualHandler::new(
                tensor,
                TensorBoolean::xor,
                TensorBooleanConst::xor_const,
                "xor",
            ))),

            // comparison ops
            "eq" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::eq,
                TensorCompareConst::eq_const,
                "eq",
            ))),
            "gt" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::gt,
                TensorCompareConst::gt_const,
                "gt",
            ))),
            "ge" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::ge,
                TensorCompareConst::ge_const,
                "ge",
            ))),
            "lt" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::lt,
                TensorCompareConst::lt_const,
                "lt",
            ))),
            "le" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::le,
                TensorCompareConst::le_const,
                "le",
            ))),
            "ne" => Some(Box::new(DualHandler::new(
                tensor,
                TensorCompare::ne,
                TensorCompareConst::ne_const,
                "ne",
            ))),

            // linear algebra
            "diagonal" => Some(Box::new(DiagonalHandler::from(tensor))),
            // "matmul" => Some(Box::new(MatMulHandler::new(tensor))),

            // reduce ops
            "max" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::max,
                    TensorReduce::max_all,
                )))
            }
            "min" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::min,
                    TensorReduce::min_all,
                )))
            }
            "product" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::product,
                    TensorReduce::product_all,
                )))
            }
            "sum" => {
                return Some(Box::new(ReduceHandler::new(
                    tensor,
                    TensorReduce::sum,
                    TensorReduce::sum_all,
                )))
            }

            // transforms
            "broadcast" => Some(Box::new(BroadcastHandler::from(tensor))),
            "cast" => Some(Box::new(CastHandler::from(tensor))),
            "expand_dims" => Some(Box::new(ExpandHandler::from(tensor))),
            "reshape" => Some(Box::new(ReshapeHandler::from(tensor))),
            "transpose" => Some(Box::new(TransposeHandler::from(tensor))),

            // trigonometry
            "asin" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::asin,
                "asin",
            ))),
            "sin" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::sin,
                "sin",
            ))),
            "sinh" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::sinh,
                "sinh",
            ))),

            "acos" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::acos,
                "acos",
            ))),
            "cos" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::cos,
                "cos",
            ))),
            "cosh" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::cosh,
                "cosh",
            ))),

            "atan" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::atan,
                "atan",
            ))),
            "tan" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::tan,
                "tan",
            ))),
            "tanh" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorTrig::tanh,
                "tanh",
            ))),

            // unary ops
            "abs" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorUnary::abs,
                "abs",
            ))),
            "all" => Some(Box::new(UnaryHandlerAsync::new(
                tensor.into(),
                TensorReduce::all,
                "all",
            ))),
            "any" => Some(Box::new(UnaryHandlerAsync::new(
                tensor.into(),
                TensorReduce::any,
                "any",
            ))),
            "exp" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorUnary::exp,
                "exp",
            ))),
            "not" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorUnaryBoolean::not,
                "not",
            ))),
            "round" => Some(Box::new(UnaryHandler::new(
                tensor.into(),
                TensorUnary::round,
                "round",
            ))),

            // views
            "dense" => {
                return Some(Box::new(SelfHandlerOwned::from(Tensor::from(
                    tensor.into_dense(),
                ))));
            }

            "sparse" => {
                return Some(Box::new(SelfHandlerOwned::from(Tensor::from(
                    tensor.into_sparse(),
                ))));
            }

            // other
            "norm" => Some(Box::new(NormHandler::from(Tensor::from(tensor)))),

            _ => None,
        }
    } else {
        None
    }
}

pub struct Static;

impl<State> Route<State> for Static
where
    State: StateInstance + From<Tensor<State::Txn, State::FE>>,
    State::FE: DenseCacheFile + AsType<Node> + Clone,
    Tensor<State::Txn, State::FE>: TryCastFrom<State>,
    Value: TryCastFrom<State>,
    Vec<Tensor<State::Txn, State::FE>>: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            return None;
        }

        match path[0].as_str() {
            "dense" => TensorType::Dense.route(&path[1..]),
            "sparse" => TensorType::Sparse.route(&path[1..]),
            "copy_from" if path.len() == 1 => Some(Box::new(CopyFromHandler)),
            "load" if path.len() == 1 => Some(Box::new(LoadHandler { class: None })),
            _ => None,
        }
    }
}

async fn constant<State>(
    txn: &State::Txn,
    shape: Shape,
    value: Number,
) -> TCResult<Dense<State::Txn, State::FE>>
where
    State: StateInstance,
    State::FE: DenseCacheFile + Clone,
{
    let store = create_dir(txn).await?;
    DenseBase::constant(store, *txn.id(), shape, value)
        .map_ok(Dense::Base)
        .await
}

async fn create_sparse<State>(
    txn: &State::Txn,
    schema: Schema,
) -> TCResult<Sparse<State::Txn, State::FE>>
where
    State: StateInstance,
{
    // let store = txn.context().create_store_unique(*txn.id()).await?;
    // SparseBase::create(*txn.id(), (schema.dtype, schema.shape.into()), store)
    //     .map_ok(Sparse::Base)
    //     .await
    Err(not_implemented!("create sparse tensor"))
}

async fn write<State, T>(tensor: T, txn_id: TxnId, key: Value, value: State) -> TCResult<()>
where
    State: StateInstance,
    T: TensorInstance
        + TensorWrite
        + TensorWriteDual<Tensor<State::Txn, State::FE>>
        + TensorTransform
        + Clone,
    Number: TryCastFrom<State>,
    Tensor<State::Txn, State::FE>: TensorInstance + TryCastFrom<State>,
{
    debug!("write {value:?} to {key}");
    let range = cast_range(tensor.shape(), key)?;

    if value.matches::<Tensor<_, _>>() {
        let value = Tensor::<_, _>::opt_cast_from(value).expect("tensor");
        tensor.write(txn_id, range, value).await
    } else if value.matches::<Number>() {
        let value = Number::opt_cast_from(value).expect("element");
        tensor.write_value(txn_id, range, value).await
    } else {
        Err(bad_request!("cannot write {value:?} to a Tensor"))
    }
}

async fn create_dir<Txn, FE>(txn: &Txn) -> TCResult<fs::Dir<FE>>
where
    Txn: Transaction<FE>,
    FE: Clone + ThreadSafe,
{
    let mut cxt = txn.context().write().await;
    let (_dir_name, dir) = cxt.create_dir_unique()?;
    fs::Dir::load(*txn.id(), dir).await
}

async fn create_tensor<State>(
    class: TensorType,
    schema: Schema,
    txn: &State::Txn,
) -> TCResult<Tensor<State::Txn, State::FE>>
where
    State: StateInstance,
    State::FE: DenseCacheFile + Clone,
{
    match class {
        TensorType::Dense => {
            constant::<State>(txn, schema.shape, schema.dtype.zero())
                .map_ok(Tensor::from)
                .await
        }
        TensorType::Sparse => {
            create_sparse::<State>(txn, schema)
                .map_ok(Tensor::from)
                .await
        }
    }
}

fn cast_bound(dim: u64, bound: Value) -> TCResult<u64> {
    let bound = i64::try_cast_from(bound, |v| TCError::unexpected(v, "an axis bound"))?;
    if bound.abs() as u64 > dim {
        return Err(bad_request!(
            "index {} is out of bounds for dimension {}",
            bound,
            dim
        ));
    }

    if bound < 0 {
        Ok(dim - bound.abs() as u64)
    } else {
        Ok(bound as u64)
    }
}

fn cast_axes(axes: Value, ndim: usize) -> TCResult<Axes> {
    debug!("cast axes {axes} with ndim {ndim}");

    match axes {
        Value::Number(x) => cast_axis(Value::Number(x), ndim).map(|x| vec![x]),
        Value::Tuple(tuple) => {
            let mut axes = Axes::with_capacity(tuple.len());
            for value in tuple {
                let x = cast_axis(value, ndim)?;
                axes.push(x);
            }
            Ok(axes)
        }
        other => Err(bad_request!("invalid axes: {other:?}")),
    }
}

fn cast_axis(axis: Value, ndim: usize) -> TCResult<usize> {
    let axis = Number::try_cast_from(axis, |v| TCError::unexpected(v, "a Tensor axis"))?;

    match axis {
        Number::UInt(axis) => Ok(axis.into()),
        Number::Int(axis) if axis < 0.into() => {
            Ok(Number::from(ndim as u64) - Number::Int(axis.abs()))
        }
        Number::Int(axis) => Ok(axis.into()),
        other => Err(TCError::unexpected(other, "a Tensor axis")),
    }
    .map(|axis: Number| axis.cast_into())
}

fn cast_axis_range<R: RangeBounds<Value> + fmt::Debug>(dim: u64, range: R) -> TCResult<AxisRange> {
    debug!("cast range from {:?} with dimension {}", range, dim);

    let start = match range.start_bound() {
        Bound::Unbounded => 0,
        Bound::Included(start) => cast_bound(dim, start.clone())?,
        Bound::Excluded(start) => cast_bound(dim, start.clone())? + 1,
    };

    let end = match range.end_bound() {
        Bound::Unbounded => dim,
        Bound::Included(end) => cast_bound(dim, end.clone())? + 1,
        Bound::Excluded(end) => cast_bound(dim, end.clone())?,
    };

    if end >= start {
        Ok(AxisRange::In(start..end, 1))
    } else {
        Err(TCError::unexpected(
            format!("{start}..{end}"),
            "axis bounds",
        ))
    }
}

pub fn cast_range(shape: &Shape, value: Value) -> TCResult<Range> {
    debug!("tensor bounds from {value} (shape is {shape:?})");

    match value {
        Value::None => Ok(Range::all(shape)),
        Value::Number(i) => {
            let bound = cast_bound(shape[0], i.into())?;
            Ok(Range::from(vec![bound]))
        }
        // Value::Tuple(range) if range.matches::<(Bound<u64>, Bound<u64>)>() => {
        //     if shape.is_empty() {
        //         return Err(bad_request!(
        //             "empty Tensor has no valid bounds, but the requested range is {}",
        //             range,
        //         ));
        //     }
        //
        //     let range = range.opt_cast_into().unwrap();
        //     Ok(Range::from(vec![cast_axis_range(shape[0], range)?]))
        // }
        Value::Tuple(bounds) => {
            if bounds.len() > shape.len() {
                return Err(bad_request!(
                    "tensor of shape {:?} does not support bounds with {} axes",
                    shape,
                    bounds.len()
                ));
            }

            let mut axis_bounds = Vec::with_capacity(shape.len());

            for (axis, bound) in bounds.into_inner().into_iter().enumerate() {
                debug!(
                    "bound for axis {} with dimension {} is {}",
                    axis, shape[axis], bound
                );

                let axis_range = if bound.is_none() {
                    AxisRange::all(shape[axis])
                } else if bound.matches::<Vec<u64>>() {
                    bound.opt_cast_into().map(AxisRange::Of).unwrap()
                } else if let Value::Number(value) = bound {
                    cast_bound(shape[axis], value.into()).map(AxisRange::At)?
                } else {
                    return Err(bad_request!("invalid bound {} for axis {}", bound, axis));
                };

                axis_bounds.push(axis_range);
            }

            Ok(Range::from(axis_bounds))
        }
        other => Err(TCError::unexpected(other, "Tensor bounds")),
    }
}

fn cast_shape(source_shape: &Shape, value: Tuple<Value>) -> TCResult<Vec<u64>> {
    if value.is_empty() {
        return Err(TCError::unexpected(value, "a Tensor shape"));
    }

    let mut shape = vec![1; value.len()];
    if value.iter().filter(|dim| *dim == &Value::None).count() > 1 {
        return Err(bad_request!(
            "Tensor reshape accepts a maximum of one unknown dimension"
        ));
    }

    let mut unknown = None;
    for (x, dim) in value.iter().enumerate() {
        match dim {
            Value::Number(dim) if dim >= &Number::from(1) => {
                shape[x] = (*dim).cast_into();
            }
            Value::None => {
                unknown = Some(x);
            }
            Value::Number(dim) if dim == &Number::from(-1) => {
                return Err(bad_request!(
                    "use value/none to specify an unknown dimension, not -1"
                ));
            }
            other => return Err(TCError::unexpected(other, "the dimension of a Tensor axis")),
        }
    }

    let size = source_shape.size();
    if let Some(unknown) = unknown {
        let known: u64 = shape.iter().product();
        if size % known == 0 {
            shape[unknown] = size / known;
        } else {
            return Err(bad_request!(
                "cannot reshape Tensor with size {} into shape {}",
                size,
                value
            ));
        }
    }

    if shape.iter().product::<u64>() == size {
        Ok(shape)
    } else {
        Err(bad_request!(
            "cannot reshape Tensor with shape {:?} into shape {}",
            source_shape,
            value
        ))
    }
}
