use futures::{TryFutureExt, TryStreamExt};

use tcgeneric::TCPathBuf;

use crate::generic::{label, PathSegment};
use crate::route::{GetHandler, Handler, PostHandler, Public, Route};
use crate::state::State;
use crate::stream::TCStream;

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
                let op = params.require(&label("op").into())?;
                let value = params.require(&label("value").into())?;
                params.expect_empty()?;

                self.source
                    .fold(txn.clone(), value, op)
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

struct Map {
    source: TCStream,
}

impl<'a> Handler<'a> for Map {
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, mut params| {
            Box::pin(async move {
                let op = params.require(&label("op").into())?;
                params.expect_empty()?;

                Ok(State::Stream(self.source.map(op)))
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
            "first" => Some(Box::new(First { source })),
            "fold" => Some(Box::new(Fold { source })),
            "for_each" => Some(Box::new(ForEach { source })),
            "map" => Some(Box::new(Map { source })),
            _ => None,
        }
    }
}
