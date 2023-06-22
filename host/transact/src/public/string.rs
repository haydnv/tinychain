use safecast::{Match, TryCastFrom, TryCastInto};

use tc_value::{TCString, Value};
use tcgeneric::{Map, PathSegment};

use super::{GetHandler, Handler, PostHandler, Route, StateInstance};

struct RenderHandler<'a> {
    template: &'a TCString,
}

impl<'a, State: StateInstance> Handler<'a, State> for RenderHandler<'a>
where
    TCString: TryCastFrom<State>,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, State::Txn, State>>
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, State::Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let params = params
                    .into_iter()
                    .map(|(id, state)| {
                        // let as_string = match state {
                        //     State::Scalar(Scalar::Value(value)) => value.to_string(),
                        //     other => format!("{:?}", other),
                        // };

                        let as_string = if state.matches::<TCString>() {
                            state.opt_cast_into().expect("string")
                        } else {
                            TCString::from(format!("{state:?}"))
                        };

                        (id, Value::String(as_string))
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

impl<State: StateInstance> Route<State> for TCString
where
    TCString: TryCastFrom<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        match path[0].as_str() {
            "render" => Some(Box::new(RenderHandler { template: self })),
            _ => None,
        }
    }
}
