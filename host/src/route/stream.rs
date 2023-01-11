use futures::{TryFutureExt, TryStreamExt};
use safecast::{Match, TryCastInto};

use tc_error::TCError;
use tc_value::Number;
use tcgeneric::TCPathBuf;

use crate::generic::{label, PathSegment};
use crate::route::{GetHandler, Handler, PostHandler, Public, Route};
use crate::state::State;
use crate::stream::{Source, TCStream};

struct Aggregate {
    source: TCStream,
}

impl<'a> Handler<'a> for Aggregate {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                key.expect_none()?;
                Ok(State::Stream(self.source.aggregate()))
            })
        }))
    }
}

struct Filter {
    source: TCStream,
}

impl<'a> Handler<'a> for Filter {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let op = params.require(&label("op").into())?;
                params.expect_empty()?;

                Ok(State::from(self.source.filter(op)))
            })
        }))
    }
}

struct First {
    source: TCStream,
}

impl<'a> Handler<'a> for First {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let mut stream = self.source.into_stream(txn.clone()).await?;
                let first = stream.try_next().map_ok(State::from).await?;
                first.get(txn, &TCPathBuf::default(), key).await
            })
        }))
    }
}

#[allow(unused)]
struct Flatten {
    source: TCStream,
}

impl<'a> Handler<'a> for Flatten {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                key.expect_none()?;
                Ok(State::Stream(self.source.flatten()))
            })
        }))
    }
}

struct Fold {
    source: TCStream,
}

impl<'a> Handler<'a> for Fold {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let item_name = params.require(&label("item_name").into())?;
                let op = params.require(&label("op").into())?;
                let value = params.require(&label("value").into())?;
                params.expect_empty()?;

                self.source
                    .fold(txn.clone(), item_name, value, op)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

struct ForEach {
    source: TCStream,
}

impl<'a> Handler<'a> for ForEach {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, mut params| {
            Box::pin(async move {
                let op = params.require(&label("op").into())?;
                params.expect_empty()?;

                self.source.for_each(txn, op).map_ok(State::from).await
            })
        }))
    }
}

impl Route for TCStream {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        let source = self.clone();
        match path[0].as_str() {
            "aggregate" => Some(Box::new(Aggregate { source })),
            "filter" => Some(Box::new(Filter { source })),
            "first" => Some(Box::new(First { source })),
            "flatten" => Some(Box::new(Flatten { source })),
            "fold" => Some(Box::new(Fold { source })),
            "for_each" => Some(Box::new(ForEach { source })),
            _ => None,
        }
    }
}

struct RangeHandler;

impl<'a> Handler<'a> for RangeHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, key| {
            Box::pin(async move {
                if key.matches::<(Number, Number, Number)>() {
                    let (start, stop, step) = key.opt_cast_into().expect("range");
                    Ok(State::Stream(TCStream::range(start, stop, step)))
                } else if key.matches::<(Number, Number)>() {
                    let (start, stop) = key.opt_cast_into().expect("range");
                    Ok(State::Stream(TCStream::range(start, stop, 1.into())))
                } else if key.matches::<Number>() {
                    let stop = key.opt_cast_into().expect("range stop");
                    Ok(State::Stream(TCStream::range(0.into(), stop, 1.into())))
                } else {
                    Err(TCError::bad_request("invalid range", key))
                }
            })
        }))
    }
}

pub(super) struct Static;

impl Route for Static {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        match path[0].as_str() {
            "range" => Some(Box::new(RangeHandler)),
            _ => None,
        }
    }
}
