use futures::TryFutureExt;

use crate::generic::{label, PathSegment};
use crate::route::{Handler, PostHandler, Route};
use crate::state::State;
use crate::stream::TCStream;

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

                self.source
                    .for_each(txn.clone(), op)
                    .map_ok(State::from)
                    .await
            })
        }))
    }
}

impl From<TCStream> for ForEach {
    fn from(source: TCStream) -> Self {
        Self { source }
    }
}

impl Route for TCStream {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        let source = self.clone();
        match path[0].as_str() {
            "for_each" => Some(Box::new(ForEach::from(source))),
            _ => None,
        }
    }
}
