use std::fmt;
use std::ops;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use collate::{Collate, Overlap, OverlapsRange, OverlapsValue};
use destream::{de, en};
use futures::TryFutureExt;
use get_size::GetSize;
use get_size_derive::*;
use safecast::{Match, TryCastFrom, TryCastInto};

use tcgeneric::{label, Id, Label, Tuple};

use super::{Value, ValueCollator};

/// The prefix of an inclusive [`Bound`]
pub const IN: Label = label("in");

/// The prefix of an exclusive [`Bound`]
pub const EX: Label = label("ex");
