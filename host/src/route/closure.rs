use tcgeneric::{Map, PathSegment};

use crate::closure::Closure;
use crate::scalar::OpDef;
use crate::state::State;

use super::{DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route};

struct ClosureHandler {
    context: Map<State>,
    op_def: OpDef,
}

impl<'a> Handler<'a> for ClosureHandler {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        let mut context = self.context;
        if let OpDef::Get((key_name, op_def)) = self.op_def {
            Some(Box::new(|txn, key| {
                Box::pin(async move {
                    context.insert(key_name, key.into());
                    OpDef::call(op_def, txn, context).await
                })
            }))
        } else {
            None
        }
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        let mut context = self.context;
        if let OpDef::Put((key_name, value_name, op_def)) = self.op_def {
            Some(Box::new(|txn, key, value| {
                Box::pin(async move {
                    context.insert(key_name, key.into());
                    context.insert(value_name, value.into());
                    OpDef::call(op_def, txn, context).await?;
                    Ok(())
                })
            }))
        } else {
            None
        }
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        let mut context = self.context;
        if let OpDef::Post(op_def) = self.op_def {
            Some(Box::new(|txn, params| {
                context.extend(params);
                Box::pin(async move { OpDef::call(op_def, txn, context).await })
            }))
        } else {
            None
        }
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        let mut context = self.context;
        if let OpDef::Delete((key_name, op_def)) = self.op_def {
            Some(Box::new(|txn, key| {
                Box::pin(async move {
                    context.insert(key_name, key.into());
                    OpDef::call(op_def, txn, context).await?;
                    Ok(())
                })
            }))
        } else {
            None
        }
    }
}

impl From<Closure> for ClosureHandler {
    fn from(closure: Closure) -> Self {
        let (context, op_def) = closure.into_inner();
        Self { context, op_def }
    }
}

impl Route for Closure {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if !path.is_empty() {
            return None;
        }

        Some(Box::new(ClosureHandler::from(self.clone())))
    }
}
