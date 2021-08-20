use std::cmp::Ordering;
use std::fmt;
use std::ops::Deref;

use async_trait::async_trait;
use bytes::Bytes;
use collate::{Collate, Collator};
use destream::{de, en};
use futures::TryFutureExt;
use handlebars::Handlebars;
use safecast::TryCastFrom;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::json;

use tc_error::*;
use tcgeneric::Id;

use super::{Link, Number};

#[derive(Clone, Default, Eq, PartialEq)]
pub struct TCString(String);

impl TCString {
    pub fn render<T: Serialize>(&self, data: T) -> TCResult<TCString> {
        Handlebars::new()
            .render_template(&self.0, &json!(data))
            .map(Self)
            .map_err(|e| TCError::bad_request("error rendering template", e))
    }
}

impl Deref for TCString {
    type Target = String;

    fn deref(&self) -> &String {
        &self.0
    }
}

impl From<String> for TCString {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<Id> for TCString {
    fn from(id: Id) -> Self {
        Self(id.to_string())
    }
}

impl From<Link> for TCString {
    fn from(link: Link) -> Self {
        Self(link.to_string())
    }
}

impl From<Number> for TCString {
    fn from(n: Number) -> Self {
        Self(n.to_string())
    }
}

impl TryCastFrom<TCString> for Bytes {
    fn can_cast_from(value: &TCString) -> bool {
        if value.ends_with('=') {
            base64::decode(&value.0).is_ok()
        } else {
            hex::decode(&value.0).is_ok()
        }
    }

    fn opt_cast_from(value: TCString) -> Option<Self> {
        if value.ends_with('=') {
            base64::decode(&value.0).ok().map(Self::from)
        } else {
            hex::decode(&value.0).ok().map(Self::from)
        }
    }
}

impl TryCastFrom<TCString> for Id {
    fn can_cast_from(value: &TCString) -> bool {
        Self::can_cast_from(&value.0)
    }

    fn opt_cast_from(value: TCString) -> Option<Self> {
        Self::opt_cast_from(value.0)
    }
}

#[async_trait]
impl de::FromStream for TCString {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        String::from_stream(cxt, decoder).map_ok(Self).await
    }
}

impl<'en> en::IntoStream<'en> for TCString {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.0.into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for TCString {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.0.to_stream(encoder)
    }
}

impl<'de> Deserialize<'de> for TCString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self)
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

#[derive(Clone, Default)]
pub struct StringCollator {
    collator: Collator<String>,
}

impl Collate for StringCollator {
    type Value = TCString;

    fn compare(&self, left: &Self::Value, right: &Self::Value) -> Ordering {
        self.collator.compare(&left.0, &right.0)
    }
}
