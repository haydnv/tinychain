use std::mem;
use std::pin::Pin;
use std::task::{self, ready};

use futures::stream::{Fuse, Stream};
use futures::StreamExt;
use ha_ndarray::{ArrayBase, CDatatype, NDArrayRead, Queue, Shape};
use pin_project::pin_project;

use tc_error::*;

#[pin_project]
pub struct BlockResize<S, T> {
    #[pin]
    source: Fuse<S>,
    shape: Shape,
    pending: Vec<T>,
    queue: Queue,
}

impl<S, T> BlockResize<S, T>
where
    S: Stream,
{
    pub fn new(source: S, block_shape: Shape) -> Result<Self, TCError> {
        let size = block_shape.iter().product();
        let context = ha_ndarray::Context::default()?;
        let queue = Queue::new(context, size)?;

        Ok(Self {
            source: source.fuse(),
            shape: block_shape,
            pending: Vec::with_capacity(size * 2),
            queue,
        })
    }
}

impl<S, A, T> Stream for BlockResize<S, T>
where
    S: Stream<Item = Result<A, TCError>>,
    A: NDArrayRead<DType = T>,
    T: CDatatype,
{
    type Item = Result<ArrayBase<Vec<T>>, TCError>;

    fn poll_next(
        self: Pin<&mut Self>,
        cxt: &mut task::Context<'_>,
    ) -> task::Poll<Option<Self::Item>> {
        let shape = self.shape.to_vec();
        let size = shape.iter().product::<usize>();
        let mut this = self.project();

        task::Poll::Ready(loop {
            if this.pending.len() > size {
                debug_assert_eq!(shape.iter().product::<usize>(), size);
                let data = this.pending.drain(..size).collect();
                let data = ArrayBase::<Vec<T>>::new(shape, data).map_err(TCError::from);
                break Some(data);
            } else {
                match ready!(this.source.as_mut().poll_next(cxt)) {
                    Some(Ok(block)) => match block.read(&this.queue) {
                        Ok(buffer) => match buffer.to_slice() {
                            Ok(slice) => this.pending.extend(slice.as_ref()),
                            Err(cause) => break Some(Err(TCError::from(cause))),
                        },
                        Err(cause) => break Some(Err(TCError::from(cause))),
                    },
                    Some(Err(cause)) => break Some(Err(cause)),
                    None if this.pending.is_empty() => break None,
                    None => {
                        let mut shape = shape;
                        let trailing_size = shape.iter().skip(1).product::<usize>();
                        shape[0] = this.pending.len() / trailing_size;
                        debug_assert_eq!(this.pending.len() % trailing_size, 0);

                        let mut data = vec![];
                        mem::swap(this.pending, &mut data);

                        let data = ArrayBase::<Vec<T>>::new(shape, data).map_err(TCError::from);
                        break Some(data);
                    }
                }
            }
        })
    }
}
