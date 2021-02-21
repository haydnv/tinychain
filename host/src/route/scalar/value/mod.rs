use tc_generic::{path_label, PathLabel, PathSegment};

use crate::route::{GetHandler, Handler, Route};
use crate::scalar::Value;

mod number;

const EQ: PathLabel = path_label(&["eq"]);

struct EqHandler<'a> {
    subject: &'a Value,
}

impl<'a> Handler<'a> for EqHandler<'a> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|_txn, key| {
            Box::pin(async move { Ok(Value::from(self.subject == &key).into()) })
        }))
    }
}

impl<'a> From<&'a Value> for EqHandler<'a> {
    fn from(subject: &'a Value) -> Self {
        Self { subject }
    }
}

impl Route for Value {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        let child_handler = match self {
            Self::Number(number) => number.route(path),
            _ => None,
        };

        if child_handler.is_some() {
            child_handler
        } else if path == &EQ[..] {
            Some(Box::new(EqHandler::from(self)))
        } else {
            None
        }
    }
}
