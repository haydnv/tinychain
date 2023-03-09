use std::fmt;

use async_trait::async_trait;
use destream::{de, en};
use safecast::{CastFrom, Match, TryCastFrom, TryCastInto};

use tc_value::{NumberType, Value, ValueType};
use tcgeneric::{Id, NativeClass};

/// The schema of a B+Tree
pub struct Schema {
    columns: Vec<Column>,
}

/// A `Column` used in the schema of a [`BTree`].
#[derive(Clone, Eq, PartialEq)]
pub struct Column {
    pub name: Id,
    pub dtype: ValueType,
    pub max_len: Option<usize>,
}

impl Column {
    /// Get the name of this column.
    #[inline]
    pub fn name(&'_ self) -> &'_ Id {
        &self.name
    }

    /// Get the [`Class`] of this column.
    #[inline]
    pub fn dtype(&self) -> ValueType {
        self.dtype
    }

    /// Get the maximum size (in bytes) of this column.
    #[inline]
    pub fn max_len(&'_ self) -> &'_ Option<usize> {
        &self.max_len
    }
}

impl<I: Into<Id>> From<(I, NumberType)> for Column {
    fn from(column: (I, NumberType)) -> Column {
        let (name, dtype) = column;
        let name: Id = name.into();
        let dtype: ValueType = dtype.into();
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType)> for Column {
    fn from(column: (Id, ValueType)) -> Column {
        let (name, dtype) = column;
        let max_len = None;

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl From<(Id, ValueType, usize)> for Column {
    fn from(column: (Id, ValueType, usize)) -> Column {
        let (name, dtype, size) = column;
        let max_len = Some(size);

        Column {
            name,
            dtype,
            max_len,
        }
    }
}

impl TryCastFrom<Value> for Column {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<(Id, ValueType)>() || value.matches::<(Id, ValueType, u64)>()
    }

    fn opt_cast_from(value: Value) -> Option<Column> {
        if value.matches::<(Id, ValueType)>() {
            let (name, dtype) = value.opt_cast_into().unwrap();

            Some(Column {
                name,
                dtype,
                max_len: None,
            })
        } else if value.matches::<(Id, ValueType, u64)>() {
            let (name, dtype, max_len) = value.opt_cast_into().unwrap();

            Some(Column {
                name,
                dtype,
                max_len: Some(max_len),
            })
        } else {
            None
        }
    }
}

impl CastFrom<Column> for Value {
    fn cast_from(column: Column) -> Self {
        Value::Tuple(
            vec![
                column.name.into(),
                column.dtype.path().into(),
                column.max_len.map(Value::from).into(),
            ]
            .into(),
        )
    }
}

struct ColumnVisitor;

#[async_trait]
impl de::Visitor for ColumnVisitor {
    type Value = Column;

    fn expecting() -> &'static str {
        "a Column definition"
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let name = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, "a Column name"))?;

        let dtype = seq
            .next_element(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(1, "a Column data type"))?;

        let max_len = seq.next_element(()).await?;

        Ok(Column {
            name,
            dtype,
            max_len,
        })
    }
}

#[async_trait]
impl de::FromStream for Column {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_seq(ColumnVisitor).await
    }
}

impl<'en> en::IntoStream<'en> for Column {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        if let Some(max_len) = self.max_len {
            (self.name, self.dtype, max_len).into_stream(encoder)
        } else {
            (self.name, self.dtype).into_stream(encoder)
        }
    }
}

impl<'en> en::ToStream<'en> for Column {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        if let Some(max_len) = self.max_len {
            en::IntoStream::into_stream((&self.name, &self.dtype, max_len), encoder)
        } else {
            en::IntoStream::into_stream((&self.name, &self.dtype), encoder)
        }
    }
}

impl<'a> From<&'a Column> for (&'a Id, ValueType) {
    fn from(col: &'a Column) -> (&'a Id, ValueType) {
        (&col.name, col.dtype)
    }
}

impl fmt::Debug for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max_len {
            Some(max_len) => write!(f, "{}: {}({})", self.name, self.dtype, max_len),
            None => write!(f, "{}: {}", self.name, self.dtype),
        }
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.max_len {
            Some(max_len) => write!(f, "{}: {}({})", self.name, self.dtype, max_len),
            None => write!(f, "{}: {}", self.name, self.dtype),
        }
    }
}
