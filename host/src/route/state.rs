use futures::future;

use safecast::{TryCastFrom, TryCastInto};

use error::*;
use generic::PathSegment;

use crate::scalar::Number;
use crate::state::State;

use super::*;

struct RootHandler<'a> {
    subject: &'a State,
}

impl<'a> Handler<'a> for RootHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        match self.subject {
            State::Tuple(tuple) => Some(Box::new(move |_txn, key| {
                let handler = async move {
                    if key.is_none() {
                        Ok(State::Tuple(tuple.clone()))
                    } else {
                        let i = Number::try_cast_from(key, |i| {
                            TCError::bad_request("Invalid tuple index", i)
                        })?;
                        let i: usize =
                            i.try_cast_into(|i| TCError::bad_request("Invalid tuple index", i))?;
                        tuple.get(i).cloned().ok_or_else(|| TCError::not_found(i))
                    }
                };

                Box::pin(handler)
            })),
            other => Some(Box::new(move |_, _| {
                Box::pin(future::ready(Ok(other.clone())))
            })),
        }
    }
}

impl Route for State {
    fn route<'a>(&'a self, path: &[PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        let child_handler = match self {
            Self::Scalar(scalar) => scalar.route(path),
            _ => None,
        };

        if let Some(handler) = child_handler {
            return Some(handler);
        }

        if path.is_empty() {
            Some(Box::new(RootHandler { subject: self }))
        } else {
            None
        }
    }
}
