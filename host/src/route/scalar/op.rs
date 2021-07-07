use std::iter;

use tcgeneric::PathSegment;

use crate::scalar::op::*;
use crate::state::State;

use crate::route::*;

struct OpHandler<'a> {
    op_def: &'a OpDef,
}

impl<'a> Handler<'a> for OpHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if let OpDef::Get((key_name, op_def)) = self.op_def.clone() {
            Some(Box::new(|txn, key| {
                Box::pin(async move {
                    let context = iter::once((key_name, State::from(key)));
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
        if let OpDef::Put((key_name, value_name, op_def)) = self.op_def.clone() {
            Some(Box::new(|txn, key, value| {
                Box::pin(async move {
                    let context = vec![(key_name, key.into()), (value_name, value)];
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
        if let OpDef::Post(op_def) = self.op_def.clone() {
            Some(Box::new(|txn, params| {
                Box::pin(async move { OpDef::call(op_def, txn, params).await })
            }))
        } else {
            None
        }
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        if let OpDef::Delete((key_name, op_def)) = self.op_def.clone() {
            Some(Box::new(|txn, key| {
                Box::pin(async move {
                    let context = iter::once((key_name, State::from(key)));
                    OpDef::call(op_def, txn, context).await?;
                    Ok(())
                })
            }))
        } else {
            None
        }
    }
}

impl Route for OpDef {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(OpHandler { op_def: self }))
        } else {
            None
        }
    }
}
