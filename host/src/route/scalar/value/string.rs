use safecast::{Match, TryCastInto};

use tc_value::{TCString, Value};
use tcgeneric::{Map, PathSegment};

use crate::route::{GetHandler, Handler, PostHandler, Route};
use crate::scalar::Scalar;
use crate::state::State;

struct RenderHandler<'a> {
    template: &'a TCString,
}

impl<'a> Handler<'a> for RenderHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, value| {
            Box::pin(async move {
                let result = if value.matches::<Map<Value>>() {
                    let data: Map<Value> = value.opt_cast_into().unwrap();
                    self.template.render(data)
                } else {
                    self.template.render(value)
                };

                result.map(Value::String).map(State::from)
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let params = params
                    .into_iter()
                    .map(|(id, state)| {
                        let as_string = match state {
                            State::Scalar(Scalar::Value(value)) => value.to_string(),
                            other => format!("{:?}", other),
                        };

                        (id, Value::String(as_string.into()))
                    })
                    .collect::<Map<Value>>();

                self.template
                    .render(params)
                    .map(Value::String)
                    .map(State::from)
            })
        }))
    }
}

impl Route for TCString {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        match path[0].as_str() {
            "render" => Some(Box::new(RenderHandler { template: self })),
            _ => None,
        }
    }
}
