use std::sync::Arc;

use crate::context::{TCContext, TCResult, TCState, TCValue};
use crate::error;
use crate::transaction::Pending;

#[derive(Hash)]
pub struct Table {
    schema: TCValue,
}

impl TCContext for Table {}

pub struct TableContext {}

impl TableContext {
    pub fn new() -> Arc<TableContext> {
        Arc::new(TableContext {})
    }
}

impl TCContext for TableContext {
    fn post(self: Arc<Self>, method: String) -> TCResult<Pending> {
        match method.as_str() {
            "new" => {
                Ok((
                    vec!["schema".to_string()],
                    Arc::new(|args| {
                        match args.get("schema") {
                        Some(TCState::Value(schema)) => Ok(Arc::new(TCState::Table(Arc::new(Table { schema: schema.clone() })))),
                        Some(other) => Err(error::bad_request("Expected vector for parameter 'schema', found", other)),
                        _ => Err(error::missing("TableContext::new requires one argument called 'schema' but found None")),
                    }
                    }),
                ))
            }
            other => Err(error::bad_request("TableContext has no such method", other)),
        }
    }
}
