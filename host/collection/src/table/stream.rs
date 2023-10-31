use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use destream::en;
use futures::{Stream, StreamExt, TryStreamExt};
use safecast::AsType;
use smallvec::SmallVec;

use tc_error::*;
use tc_transact::{IntoView, Transaction};
use tcgeneric::{TCBoxTryStream, ThreadSafe};

use super::{Node, Range, Row, Table, TableInstance, TableSchema, TableStream};

type PermitRead = tc_transact::lock::PermitRead<Range>;

/// A stream over a range of rows in a `Table`
pub struct Rows<'a> {
    permit: PermitRead,
    rows: TCBoxTryStream<'a, Row>,
}

impl<'a> Rows<'a> {
    pub(super) fn new(permit: PermitRead, keys: TCBoxTryStream<'a, Row>) -> Self {
        Self { permit, rows: keys }
    }

    pub(super) fn limit(self, limit: usize) -> Self {
        Self {
            permit: self.permit,
            rows: Box::pin(self.rows.take(limit)),
        }
    }

    pub(super) fn select(self, schema: TableSchema, selection_schema: TableSchema) -> Self {
        let mut indices = SmallVec::<[Option<usize>; 32]>::with_capacity(selection_schema.len());

        for col_name in selection_schema.columns() {
            if let Some(mut index) = schema.columns().position(|c| col_name == c) {
                index -= indices.iter().filter(|i| (**i) < Some(index)).count();
                indices.push(Some(index));
            } else {
                indices.push(None);
            }
        }

        let rows = self.rows.map_ok(move |mut row| {
            indices
                .iter()
                .copied()
                .map(|i| i.map(|i| row.remove(i)).unwrap_or_default())
                .collect()
        });

        Self {
            permit: self.permit,
            rows: Box::pin(rows),
        }
    }
}

impl<'a> Stream for Rows<'a> {
    type Item = TCResult<Row>;

    fn poll_next(mut self: Pin<&mut Self>, cxt: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rows).poll_next(cxt)
    }
}

pub struct TableView<'en> {
    schema: TableSchema,
    rows: Rows<'en>,
}

impl<'en> TableView<'en> {
    fn new(schema: TableSchema, rows: Rows<'en>) -> Self {
        Self { schema, rows }
    }
}

impl<'en> en::IntoStream<'en> for TableView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        (self.schema, en::SeqStream::from(self.rows)).into_stream(encoder)
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for Table<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: AsType<Node> + ThreadSafe,
{
    type Txn = Txn;
    type View = TableView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        let schema = self.schema().clone();
        let rows = self.rows(*txn.id()).await?;
        Ok(TableView::new(schema, rows))
    }
}
