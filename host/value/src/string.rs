//! A TinyChain String

use std::cmp::Ordering;
use std::fmt;
use std::mem::size_of;
use std::sync::Arc;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD_NO_PAD;
use base64::Engine;
use bytes::Bytes;
use collate::{Collate, Collator};
use destream::{de, en};
use futures::TryFutureExt;
use get_size::GetSize;
use handlebars::Handlebars;
use safecast::TryCastFrom;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::json;

use tc_error::*;
use tcgeneric::Id;

use super::{Link, Number};

/// A TinyChain String
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct TCString(Arc<str>);

impl GetSize for TCString {
    fn get_size(&self) -> usize {
        size_of::<Arc<str>>() + self.0.as_bytes().len()
    }
}

impl Default for TCString {
    fn default() -> Self {
        Self::from(String::default())
    }
}

impl TCString {
    /// Render this string as a [`Handlebars`] template with the given `data`.
    ///
    /// Example:
    /// ```
    /// # use std::collections::HashMap;
    /// # use tc_value::TCString;
    /// let data: HashMap<_, _> = std::iter::once(("name", "world")).collect();
    /// assert_eq!(
    ///     TCString::from("Hello, {{name}}!".to_string()).render(data).unwrap().as_str(),
    ///     "Hello, world!");
    /// ```
    ///
    /// See the [`handlebars`] documentation for a complete description of the formatting options.
    pub fn render<T: Serialize>(&self, data: T) -> TCResult<TCString> {
        Handlebars::new()
            .render_template(self.0.as_ref(), &json!(data))
            .map(Self::from)
            .map_err(|cause| bad_request!("template render error").consume(cause))
    }

    /// Borrow this [`TCString`] as a `str`.
    pub fn as_str(&self) -> &str {
        self.0.as_ref()
    }
}

impl PartialEq<Id> for TCString {
    fn eq(&self, other: &Id) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<String> for TCString {
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<str> for TCString {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl From<String> for TCString {
    fn from(s: String) -> Self {
        Self(s.into())
    }
}

impl From<Id> for TCString {
    fn from(id: Id) -> Self {
        Self(id.into_inner())
    }
}

impl From<Link> for TCString {
    fn from(link: Link) -> Self {
        Self::from(link.to_string())
    }
}

impl From<Number> for TCString {
    fn from(n: Number) -> Self {
        Self::from(n.to_string())
    }
}

impl TryCastFrom<TCString> for Bytes {
    fn can_cast_from(value: &TCString) -> bool {
        if value.as_str().ends_with('=') {
            STANDARD_NO_PAD.decode(value.as_str()).is_ok()
        } else {
            hex::decode(value.as_str()).is_ok()
        }
    }

    fn opt_cast_from(value: TCString) -> Option<Self> {
        if value.as_str().ends_with('=') {
            STANDARD_NO_PAD.decode(value.as_str()).ok().map(Self::from)
        } else {
            hex::decode(value.as_str()).ok().map(Self::from)
        }
    }
}

#[async_trait]
impl de::FromStream for TCString {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        String::from_stream(cxt, decoder).map_ok(Self::from).await
    }
}

impl<'en> en::IntoStream<'en> for TCString {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_str(self.as_str())
    }
}

impl<'en> en::ToStream<'en> for TCString {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        encoder.encode_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for TCString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::from)
    }
}

impl Serialize for TCString {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl fmt::Debug for TCString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl fmt::Display for TCString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A [`Collator`] for [`TCString`] values.
#[derive(Clone, Default, Eq, PartialEq)]
pub struct StringCollator {
    collator: Collator<Arc<str>>,
}

impl Collate for StringCollator {
    type Value = TCString;

    fn cmp(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        self.collator.cmp(&left.0, &right.0)
    }
}
