use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use crate::context::{TCContext, TCResult, TCState, TCValue};
use crate::error;
use crate::host::Host;
use crate::state::chain::ChainContext;
use crate::transaction::Transaction;

#[derive(Hash)]
pub struct Table {
    schema: Vec<(TCValue, TCValue)>,
}

impl TCContext for Table {}

pub struct TableContext {
    chain_context: Arc<ChainContext>,
}

impl TableContext {
    pub fn new(chain_context: Arc<ChainContext>) -> Arc<TableContext> {
        Arc::new(TableContext { chain_context })
    }

    async fn new_table(host: Arc<Host>, schema: TCValue) -> TCResult<Table> {
        let mut valid_columns: Vec<(TCValue, TCValue)> = vec![];

        let columns = TCValue::vector(&schema)?;
        for column in columns {
            let column = TCValue::vector(&column)?;
            if column.len() != 2 {
                return Err(error::bad_request(
                    "Expected a column name and type but found",
                    format!("{:?}", column),
                ));
            }

            match &column[..2] {
                [TCValue::r#String(name), TCValue::Link(datatype)] => {
                    valid_columns.push((
                        TCValue::r#String(name.clone()),
                        TCValue::Link(datatype.clone()),
                    ));
                }
                other => {
                    return Err(error::bad_request(
                        "Expected a column definition but found",
                        format!("{:?}", other),
                    ));
                }
            }
        }

        let mut data_types: Vec<String> = vec![];
        for (_, link) in &valid_columns {
            data_types.push(TCValue::link_string(&link)?);
        }
        let data_types = data_types.iter().map(|d| host.clone().get(d.clone()));
        for result in join_all(data_types).await {
            match result {
                Ok(_) => (),
                Err(cause) => {
                    return Err(cause);
                }
            }
        }

        Ok(Table {
            schema: valid_columns,
        })
    }
}

#[async_trait]
impl TCContext for TableContext {
    async fn post(
        self: Arc<Self>,
        host: Arc<Host>,
        method: String,
        txn: Arc<Transaction>,
    ) -> TCResult<Arc<TCState>> {
        if method != "new" {
            return Err(error::not_found(method));
        }

        if let TCState::Value(schema) = &*txn.require("schema")? {
            Ok(Arc::new(TCState::Table(Arc::new(
                Self::new_table(host, schema.clone()).await?,
            ))))
        } else {
            Err(error::bad_request(
                "TableContext::new takes one parameter",
                "schema",
            ))
        }
    }
}
