//! Immutable values which always reside entirely in memory

use std::collections::HashSet;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::ops::{Bound, Deref, DerefMut};
use std::str::FromStr;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use bytes::Bytes;
use destream::{de, en};
use futures::future::TryFutureExt;
use futures::stream::{self, StreamExt, TryStreamExt};
use get_size::GetSize;
use log::{debug, warn};
use safecast::{as_type, Match, TryCastFrom, TryCastInto};
use smallvec::SmallVec;

use tc_error::*;
use tc_transact::public::{Public, StateInstance, ToState};
use tc_value::{Float, Host, Link, Number, TCString, Value, ValueType};
use tcgeneric::*;

pub use op::*;
pub use reference::*;
pub use tc_value as value;

pub mod op;
pub mod public;
pub mod reference;

const ERR_NO_SELF: &str = "Op context has no $self--consider using a method instead";
const PREFIX: PathLabel = path_label(&["state", "scalar"]);

/// The label of an instance in its own method context
pub const SELF: Label = label("self");

/// The [`Class`] of a [`Scalar`].
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ScalarType {
    Cluster,
    Map,
    Op(OpDefType),
    Range,
    Ref(RefType),
    Tuple,
    Value(ValueType),
}

impl Class for ScalarType {}

impl NativeClass for ScalarType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        debug!("ScalarType::from_path {}", TCPath::from(path));
        if path.len() >= 2 && &path[..2] != &PREFIX[..] {
            return None;
        }

        if path.len() == 3 {
            match path[2].as_str() {
                "cluster" => Some(Self::Cluster),
                "map" => Some(Self::Map),
                "range" => Some(Self::Range),
                "tuple" => Some(Self::Tuple),
                _ => None,
            }
        } else if path.len() > 2 {
            match path[2].as_str() {
                "op" => OpDefType::from_path(path).map(Self::Op),
                "ref" => RefType::from_path(path).map(Self::Ref),
                "value" => ValueType::from_path(path).map(Self::Value),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        let prefix = TCPathBuf::from(PREFIX);

        match self {
            Self::Cluster => prefix.append(label("cluster")),
            Self::Map => prefix.append(label("map")),
            Self::Op(odt) => odt.path(),
            Self::Range => prefix.append(label("range")),
            Self::Ref(rt) => rt.path(),
            Self::Value(vt) => vt.path(),
            Self::Tuple => prefix.append(label("tuple")),
        }
    }
}

impl Default for ScalarType {
    fn default() -> Self {
        Self::Value(ValueType::default())
    }
}

impl From<ValueType> for ScalarType {
    fn from(vt: ValueType) -> Self {
        Self::Value(vt)
    }
}

impl fmt::Debug for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Cluster => f.write_str("Cluster reference"),
            Self::Map => f.write_str("Map<Scalar>"),
            Self::Op(odt) => fmt::Debug::fmt(odt, f),
            Self::Range => f.write_str("Range"),
            Self::Ref(rt) => fmt::Debug::fmt(rt, f),
            Self::Value(vt) => fmt::Debug::fmt(vt, f),
            Self::Tuple => f.write_str("Tuple<Scalar>"),
        }
    }
}

/// A reference to a `Cluster`
#[derive(Clone, Eq, PartialEq)]
pub struct ClusterRef(TCPathBuf);

impl GetSize for ClusterRef {
    fn get_size(&self) -> usize {
        self.0.get_size()
    }
}

impl ClusterRef {
    /// Consume this reference and return its `TCPathBuf`
    pub fn into_path(self) -> TCPathBuf {
        self.0
    }

    /// Borrow this reference's `TCPathBuf`
    pub fn path(&self) -> &TCPathBuf {
        &self.0
    }
}

impl<'a, D: Digest> Hash<D> for &'a ClusterRef {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self.0)
    }
}

impl From<TCPathBuf> for ClusterRef {
    fn from(path: TCPathBuf) -> Self {
        Self(path)
    }
}

impl fmt::Debug for ClusterRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "reference to Cluster at {:?}", self.0)
    }
}

impl fmt::Display for ClusterRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "reference to Cluster at {}", self.0)
    }
}

/// A scalar value, i.e. an immutable state always held in main memory and never split into blocks.
#[derive(Clone, Eq, PartialEq)]
pub enum Scalar {
    Cluster(ClusterRef),
    Map(Map<Self>),
    Op(OpDef),
    Range((Bound<Value>, Bound<Value>)),
    Ref(Box<TCRef>),
    Tuple(Tuple<Self>),
    Value(Value),
}

as_type!(Scalar, Cluster, ClusterRef);
as_type!(Scalar, Map, Map<Scalar>);
as_type!(Scalar, Op, OpDef);
as_type!(Scalar, Range, (Bound<Value>, Bound<Value>));
as_type!(Scalar, Ref, Box<TCRef>);
as_type!(Scalar, Tuple, Tuple<Scalar>);
as_type!(Scalar, Value, Value);

impl Scalar {
    /// Return `true` if this `Scalar` variant is a [`Map`].
    pub fn is_map(&self) -> bool {
        match self {
            Self::Map(_) => true,
            _ => false,
        }
    }

    /// Return `true` if self is an empty tuple, default link, or `Value::None`.
    pub fn is_none(&self) -> bool {
        match self {
            Self::Map(map) => map.is_empty(),
            Self::Tuple(tuple) => tuple.is_empty(),
            Self::Value(value) => value.is_none(),
            _ => false,
        }
    }

    /// Return `true` if this `Scalar` variant is a [`Tuple`].
    pub fn is_tuple(&self) -> bool {
        match self {
            Self::Tuple(_) => true,
            Self::Value(value) => value.is_tuple(),
            _ => false,
        }
    }

    /// Cast this `Scalar` into the specified [`ScalarType`], if possible.
    pub fn into_type(self, class: ScalarType) -> Option<Self> {
        debug!("cast into {:?} from {:?}: {:?}", class, self.class(), self);

        if self.class() == class {
            return Some(self);
        }

        use OpDefType as ODT;
        use OpRefType as ORT;
        use RefType as RT;
        use ScalarType as ST;

        match class {
            ST::Cluster => self.opt_cast_into().map(ClusterRef).map(Self::Cluster),
            ST::Map => self.opt_cast_into().map(Self::Map),
            ST::Op(odt) => match odt {
                ODT::Get => self.opt_cast_into().map(OpDef::Get).map(Self::Op),

                ODT::Put => self.opt_cast_into().map(OpDef::Put).map(Self::Op),

                ODT::Post => self.opt_cast_into().map(OpDef::Post).map(Self::Op),

                ODT::Delete => self.opt_cast_into().map(OpDef::Delete).map(Self::Op),
            },
            ST::Range => self.opt_cast_into().map(Self::Range),
            ST::Ref(rt) => match rt {
                RT::After => self
                    .opt_cast_into()
                    .map(Box::new)
                    .map(TCRef::After)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::Case => self
                    .opt_cast_into()
                    .map(Box::new)
                    .map(TCRef::Case)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::Id => self
                    .opt_cast_into()
                    .map(TCRef::Id)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::If => self
                    .opt_cast_into()
                    .map(Box::new)
                    .map(TCRef::If)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::Op(ort) => {
                    debug!("cast into op ref from {:?}", self);

                    if ort == ORT::Delete && self.matches::<(Scalar, Scalar)>() {
                        debug!("cast into DELETE ref from {:?}", self);

                        let (subject, key): (Scalar, Scalar) = self.opt_cast_into().unwrap();
                        let delete_ref: Option<(Subject, Scalar)> = match subject {
                            Scalar::Ref(tc_ref) => match *tc_ref {
                                TCRef::Id(id_ref) => Some((id_ref.into(), key)),
                                TCRef::Op(OpRef::Get((subject, none))) if none.is_none() => {
                                    Some((subject, key))
                                }
                                other => {
                                    debug!("invalid DELETE subject: {:?}", other);
                                    None
                                }
                            },
                            other => {
                                debug!("invalid DELETE subject: {:?}", other);
                                None
                            }
                        };

                        if let Some((subject, key)) = &delete_ref {
                            debug!("DELETE ref is {}: {:?}", subject, key);
                        } else {
                            debug!("could not cast into DELETE ref");
                        }

                        delete_ref.map(OpRef::Delete).map(TCRef::Op).map(Self::from)
                    } else if self.matches::<Tuple<Scalar>>() {
                        let tuple: Tuple<Scalar> = self.opt_cast_into().unwrap();
                        debug!("cast into {:?} from tuple {:?}", ort, tuple);

                        let op_ref = match ort {
                            ORT::Get => tuple.opt_cast_into().map(OpRef::Get),
                            ORT::Put => tuple.opt_cast_into().map(OpRef::Put),
                            ORT::Post => {
                                debug!("subject is {:?} (a {:?})", &tuple[0], tuple[0].class());
                                tuple.opt_cast_into().map(OpRef::Post)
                            }
                            ORT::Delete => tuple.opt_cast_into().map(OpRef::Delete),
                        };

                        op_ref.map(TCRef::Op).map(Self::from)
                    } else {
                        debug!("cannot cast into {:?} (not a tuple)", ort);
                        None
                    }
                }

                RT::While => self
                    .opt_cast_into()
                    .map(Box::new)
                    .map(TCRef::While)
                    .map(Box::new)
                    .map(Scalar::Ref),

                RT::With => self
                    .opt_cast_into()
                    .map(Box::new)
                    .map(TCRef::With)
                    .map(Box::new)
                    .map(Scalar::Ref),
            },

            ST::Value(vt) => Value::opt_cast_from(self)
                .and_then(|value| value.into_type(vt))
                .map(Scalar::Value),

            ST::Tuple => match self {
                Self::Map(map) => Some(Self::Tuple(map.into_iter().map(|(_, v)| v).collect())),
                Self::Tuple(tuple) => Some(Self::Tuple(tuple)),
                _ => None,
            },
        }
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self::Value(Value::default())
    }
}

impl Instance for Scalar {
    type Class = ScalarType;

    fn class(&self) -> ScalarType {
        use ScalarType as ST;
        match self {
            Self::Cluster(_) => ST::Cluster,
            Self::Map(_) => ST::Map,
            Self::Op(op) => ST::Op(op.class()),
            Self::Range(_) => ST::Range,
            Self::Ref(tc_ref) => ST::Ref(tc_ref.class()),
            Self::Tuple(_) => ST::Tuple,
            Self::Value(value) => ST::Value(value.class()),
        }
    }
}

impl GetSize for Scalar {
    fn get_size(&self) -> usize {
        match self {
            Self::Cluster(cluster) => cluster.get_size(),
            Self::Map(map) => map.get_size(),
            Self::Op(op) => op.get_size(),
            Self::Range((start, end)) => start.get_size() + end.get_size(),
            Self::Ref(tc_ref) => tc_ref.get_size(),
            Self::Tuple(tuple) => tuple.get_size(),
            Self::Value(value) => value.get_size(),
        }
    }
}

#[async_trait]
impl<State> Refer<State> for Scalar
where
    State: StateInstance + Refer<State> + From<Scalar>,
    State::Closure: From<(Map<State>, OpDef)> + TryCastFrom<State>,
    Map<State>: TryFrom<State, Error = TCError>,
    Value: TryFrom<State, Error = TCError> + TryCastFrom<State>,
    bool: TryCastFrom<State>,
{
    fn dereference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Map(map) => {
                let map = map
                    .into_iter()
                    .map(|(id, scalar)| (id, scalar.dereference_self(path)))
                    .collect();

                Self::Map(map)
            }
            Self::Op(op_def) => Self::Op(op_def.dereference_self(path)),
            Self::Ref(tc_ref) => {
                let tc_ref = tc_ref.dereference_self(path);
                Self::Ref(Box::new(tc_ref))
            }
            Self::Tuple(tuple) => {
                let tuple = tuple
                    .into_iter()
                    .map(|scalar| scalar.dereference_self(path))
                    .collect();

                Self::Tuple(tuple)
            }
            other => other,
        }
    }

    fn is_conditional(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(|scalar| scalar.is_conditional()),
            Self::Ref(tc_ref) => tc_ref.is_conditional(),
            Self::Tuple(tuple) => tuple.iter().any(|scalar| scalar.is_conditional()),
            _ => false,
        }
    }

    fn is_inter_service_write(&self, cluster_path: &[PathSegment]) -> bool {
        match self {
            Self::Map(map) => map
                .values()
                .any(|scalar| scalar.is_inter_service_write(cluster_path)),

            Self::Op(op_def) => op_def.is_inter_service_write(cluster_path),

            Self::Ref(tc_ref) => tc_ref.is_inter_service_write(cluster_path),

            Self::Tuple(tuple) => tuple
                .iter()
                .any(|scalar| scalar.is_inter_service_write(cluster_path)),

            _ => false,
        }
    }

    fn is_ref(&self) -> bool {
        match self {
            Self::Map(map) => map.values().any(Self::is_ref),
            Self::Ref(_) => true,
            Self::Tuple(tuple) => tuple.iter().any(Self::is_ref),
            _ => false,
        }
    }

    fn reference_self(self, path: &TCPathBuf) -> Self {
        match self {
            Self::Map(map) => {
                let map = map
                    .into_iter()
                    .map(|(id, scalar)| (id, scalar.reference_self(path)))
                    .collect();

                Self::Map(map)
            }
            Self::Op(op_def) => {
                let before = op_def.clone();
                let after = op_def.reference_self(path);
                if before == after {
                    Self::Op(after)
                } else {
                    Self::Ref(Box::new(TCRef::With(Box::new(With::new(
                        vec![SELF.into()].into(),
                        after,
                    )))))
                }
            }
            Self::Ref(tc_ref) => {
                let tc_ref = tc_ref.reference_self(path);
                Self::Ref(Box::new(tc_ref))
            }
            Self::Tuple(tuple) => {
                let tuple = tuple
                    .into_iter()
                    .map(|scalar| scalar.reference_self(path))
                    .collect();

                Self::Tuple(tuple)
            }
            other => other,
        }
    }

    fn requires(&self, deps: &mut HashSet<Id>) {
        match self {
            Self::Map(map) => {
                for scalar in map.values() {
                    scalar.requires(deps);
                }
            }
            Self::Ref(tc_ref) => tc_ref.requires(deps),
            Self::Tuple(tuple) => {
                for scalar in tuple.iter() {
                    scalar.requires(deps);
                }
            }
            _ => {}
        }
    }

    async fn resolve<'a, T: ToState<State> + Public<State> + Instance>(
        self,
        context: &'a Scope<'a, State, T>,
        txn: &'a State::Txn,
    ) -> TCResult<State> {
        debug!("Scalar::resolve {:?}", self);

        match self {
            Self::Map(map) => {
                let mut resolved = stream::iter(map)
                    .map(|(id, scalar)| scalar.resolve(context, txn).map_ok(|state| (id, state)))
                    .buffer_unordered(num_cpus::get());

                let mut map = Map::new();
                while let Some((id, state)) = resolved.try_next().await? {
                    map.insert(id, state);
                }

                Ok(State::from(map))
            }
            Self::Ref(tc_ref) => tc_ref.resolve(context, txn).await,
            Self::Tuple(tuple) => {
                let len = tuple.len();
                let mut resolved = stream::iter(tuple)
                    .map(|scalar| scalar.resolve(context, txn))
                    .buffered(num_cpus::get());

                let mut tuple = Vec::with_capacity(len);
                while let Some(state) = resolved.try_next().await? {
                    tuple.push(state);
                }

                Ok(State::from(Tuple::from(tuple)))
            }
            other => Ok(State::from(other)),
        }
    }
}

impl<D: Digest> Hash<D> for Scalar {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(&self)
    }
}

impl<'a, D: Digest> Hash<D> for &'a Scalar {
    fn hash(self) -> Output<D> {
        match self {
            Scalar::Cluster(cluster_ref) => Hash::<D>::hash(cluster_ref),
            Scalar::Map(map) => Hash::<D>::hash(map.deref()),
            Scalar::Op(op) => Hash::<D>::hash(op),
            Scalar::Range(_range) => todo!("hash a range"),
            Scalar::Ref(tc_ref) => Hash::<D>::hash(tc_ref.deref()),
            Scalar::Tuple(tuple) => Hash::<D>::hash(tuple.deref()),
            Scalar::Value(value) => Hash::<D>::hash(value),
        }
    }
}

impl From<Id> for Scalar {
    fn from(id: Id) -> Self {
        Self::Value(id.into())
    }
}

impl From<IdRef> for Scalar {
    fn from(id_ref: IdRef) -> Self {
        Self::Ref(Box::new(TCRef::Id(id_ref)))
    }
}

impl From<Host> for Scalar {
    fn from(host: Host) -> Self {
        Value::from(host).into()
    }
}

impl From<Link> for Scalar {
    fn from(link: Link) -> Self {
        Value::from(link).into()
    }
}

impl From<Number> for Scalar {
    fn from(n: Number) -> Self {
        Self::Value(n.into())
    }
}

impl From<OpRef> for Scalar {
    fn from(op_ref: OpRef) -> Self {
        Self::Ref(Box::new(TCRef::Op(op_ref)))
    }
}

impl From<TCPathBuf> for Scalar {
    fn from(path: TCPathBuf) -> Self {
        Self::Value(path.into())
    }
}

impl From<TCRef> for Scalar {
    fn from(tc_ref: TCRef) -> Self {
        Self::Ref(Box::new(tc_ref))
    }
}

impl From<Tuple<Value>> for Scalar {
    fn from(tuple: Tuple<Value>) -> Self {
        Self::Value(tuple.into())
    }
}

impl From<bool> for Scalar {
    fn from(b: bool) -> Self {
        Self::Value(b.into())
    }
}

impl From<i64> for Scalar {
    fn from(n: i64) -> Self {
        Self::Value(n.into())
    }
}

impl From<usize> for Scalar {
    fn from(n: usize) -> Self {
        Self::Value(n.into())
    }
}

impl From<u64> for Scalar {
    fn from(n: u64) -> Self {
        Self::Value(n.into())
    }
}

impl TryFrom<Scalar> for bool {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> TCResult<Self> {
        match scalar {
            Scalar::Value(value) => value.try_into(),
            other => Err(TCError::unexpected(other, "a boolean")),
        }
    }
}

impl TryFrom<Scalar> for Id {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> TCResult<Self> {
        match scalar {
            Scalar::Value(value) => value.try_into(),
            other => Err(TCError::unexpected(other, "a boolean")),
        }
    }
}

impl TryFrom<Scalar> for Map<Scalar> {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> TCResult<Map<Scalar>> {
        match scalar {
            Scalar::Map(map) => Ok(map),

            Scalar::Tuple(tuple) => tuple
                .into_iter()
                .map(|item| -> TCResult<(Id, Scalar)> { item.try_into() })
                .collect(),

            other => Err(TCError::unexpected(other, "a Map")),
        }
    }
}

impl TryFrom<Scalar> for Map<Value> {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> TCResult<Map<Value>> {
        debug!("try to construct a Map<Value> from {scalar:?}");

        match scalar {
            Scalar::Map(map) => map
                .into_iter()
                .map(|(id, scalar)| Value::try_from(scalar).map(|value| (id, value)))
                .collect(),

            Scalar::Tuple(tuple) => tuple
                .into_iter()
                .map(|item| -> TCResult<(Id, Value)> { item.try_into() })
                .collect(),

            Scalar::Value(value) => value.try_into(),

            other => Err(TCError::unexpected(other, "a Map")),
        }
    }
}

impl TryFrom<Scalar> for Tuple<Scalar> {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        match scalar {
            Scalar::Map(map) => Ok(map
                .into_iter()
                .map(|(id, scalar)| Scalar::Tuple(vec![Scalar::from(id), scalar].into()))
                .collect()),

            Scalar::Tuple(tuple) => Ok(tuple),

            Scalar::Value(value) => {
                let tuple = Tuple::<Value>::try_from(value)?;
                Ok(tuple.into_iter().map(Scalar::Value).collect())
            }

            other => Err(TCError::unexpected(other, "a Tuple")),
        }
    }
}

impl<T: TryFrom<Scalar>> TryFrom<Scalar> for (Id, T)
where
    TCError: From<T::Error>,
{
    type Error = TCError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        match scalar {
            Scalar::Tuple(mut tuple) if tuple.len() == 2 => {
                let value = tuple.pop().expect("value");
                let key = tuple.pop().expect("key");
                Ok((Id::try_from(key)?, value.try_into()?))
            }
            other => Err(TCError::unexpected(other, "a map item")),
        }
    }
}

impl TryFrom<Scalar> for TCRef {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> Result<Self, Self::Error> {
        match scalar {
            Scalar::Ref(tc_ref) => Ok(*tc_ref),
            other => Err(TCError::unexpected(other, "a reference")),
        }
    }
}

impl TryFrom<Scalar> for Value {
    type Error = TCError;

    fn try_from(scalar: Scalar) -> TCResult<Value> {
        match scalar {
            Scalar::Value(value) => Ok(value),
            Scalar::Tuple(tuple) => tuple.into_iter().map(Value::try_from).collect(),
            other => Err(TCError::unexpected(other, "a Value")),
        }
    }
}

impl TryCastFrom<Scalar> for (Bound<Value>, Bound<Value>) {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Range(_) => true,
            Scalar::Value(value) => Self::can_cast_from(value),
            Scalar::Tuple(tuple) if tuple.len() == 2 => tuple.iter().all(|scalar| match scalar {
                Scalar::Value(value) => Bound::<Value>::can_cast_from(value),
                Scalar::Tuple(tuple) if tuple.len() == 2 => match &tuple[0] {
                    Scalar::Value(Value::String(bound)) if bound == "in" || bound == "ex" => {
                        Value::can_cast_from(&tuple[1])
                    }
                    _ => false,
                },
                _ => false,
            }),
            Scalar::Tuple(_) => false,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Range(range) => Some(range),
            Scalar::Value(value) => Self::opt_cast_from(value),
            Scalar::Tuple(mut tuple) if tuple.len() == 2 => {
                let hi: Value = tuple.pop().expect("hi").opt_cast_into()?;
                let lo: Value = tuple.pop().expect("lo").opt_cast_into()?;
                Some((lo.opt_cast_into()?, hi.opt_cast_into()?))
            }
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for OpDef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Op(_) => true,
            Scalar::Tuple(tuple) => {
                GetOp::can_cast_from(tuple)
                    || PutOp::can_cast_from(tuple)
                    || PostOp::can_cast_from(tuple)
                    || DeleteOp::can_cast_from(tuple)
            }
            Scalar::Value(Value::Tuple(tuple)) => {
                GetOp::can_cast_from(tuple)
                    || PutOp::can_cast_from(tuple)
                    || PostOp::can_cast_from(tuple)
                    || DeleteOp::can_cast_from(tuple)
            }
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Op(op_def) => Some(op_def),
            Scalar::Tuple(tuple) => {
                if PutOp::can_cast_from(&tuple) {
                    tuple.opt_cast_into().map(Self::Put)
                } else if GetOp::can_cast_from(&tuple) {
                    tuple.opt_cast_into().map(Self::Get)
                } else if PostOp::can_cast_from(&tuple) {
                    tuple.opt_cast_into().map(Self::Post)
                } else if DeleteOp::can_cast_from(&tuple) {
                    tuple.opt_cast_into().map(Self::Delete)
                } else {
                    None
                }
            }
            Scalar::Value(Value::Tuple(tuple)) => {
                Scalar::Tuple(tuple.into_iter().collect()).opt_cast_into()
            }
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for TCRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => Some(*tc_ref),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for IdRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => match &**tc_ref {
                TCRef::Id(_) => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => match *tc_ref {
                TCRef::Id(id_ref) => Some(id_ref),
                _ => None,
            },
            _ => None,
        }
    }
}

impl<T: Clone + TryCastFrom<Scalar>> TryCastFrom<Scalar> for Map<T> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Map(map) => map.values().all(T::can_cast_from),
            Scalar::Tuple(tuple) => Vec::<(Id, T)>::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Map(map) => {
                let mut dest = Map::new();
                for (key, value) in map.into_iter() {
                    let value = T::opt_cast_from(value)?;
                    dest.insert(key, value);
                }
                Some(dest)
            }
            Scalar::Tuple(tuple) => {
                if let Some(entries) = Vec::<(Id, T)>::opt_cast_from(tuple) {
                    Some(entries.into_iter().collect())
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Tuple<Scalar> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(_) => true,
            Scalar::Value(Value::Tuple(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Some(tuple),
            Scalar::Value(Value::Tuple(tuple)) => Some(tuple.into_iter().collect()),
            _ => None,
        }
    }
}

impl<const N: usize, T> TryCastFrom<Scalar> for SmallVec<[T; N]>
where
    T: TryCastFrom<Scalar>,
    [T; N]: smallvec::Array<Item = T>,
{
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for Vec<T> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Tuple<Value> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => tuple.iter().all(Value::can_cast_from),
            Scalar::Value(Value::Tuple(_)) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => tuple.into_iter().map(Value::opt_cast_from).collect(),
            Scalar::Value(Value::Tuple(tuple)) => Some(tuple),
            _ => None,
        }
    }
}

macro_rules! from_value {
    ($t:ty) => {
        impl TryCastFrom<Scalar> for $t {
            fn can_cast_from(scalar: &Scalar) -> bool {
                match scalar {
                    Scalar::Value(value) => Self::can_cast_from(value),
                    _ => false,
                }
            }

            fn opt_cast_from(scalar: Scalar) -> Option<Self> {
                match scalar {
                    Scalar::Value(value) => Self::opt_cast_from(value),
                    _ => None,
                }
            }
        }
    };
}

from_value!(Bytes);
from_value!(Float);
from_value!(Link);
from_value!(Host);
from_value!(Number);
from_value!(TCPathBuf);
from_value!(TCString);
from_value!(bool);
from_value!(usize);
from_value!(u64);

impl TryCastFrom<Scalar> for Value {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => tuple.iter().all(Self::can_cast_from),
            Scalar::Value(_) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => {
                let mut value = Vec::with_capacity(tuple.len());
                for item in tuple.into_iter() {
                    if let Some(item) = Self::opt_cast_from(item) {
                        value.push(item);
                    } else {
                        return None;
                    }
                }

                Some(Value::Tuple(value.into()))
            }
            Scalar::Value(value) => Some(value),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for Id {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Value(value) => Self::can_cast_from(value),
            Scalar::Ref(tc_ref) => Self::can_cast_from(&**tc_ref),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Value(value) => Self::opt_cast_from(value),
            Scalar::Ref(tc_ref) => Self::opt_cast_from(*tc_ref),
            _ => None,
        }
    }
}

impl TryCastFrom<Scalar> for OpRef {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Ref(tc_ref) => Self::can_cast_from(&**tc_ref),
            get_ref if GetRef::can_cast_from(get_ref) => true,
            put_ref if PutRef::can_cast_from(put_ref) => true,
            post_ref if PostRef::can_cast_from(post_ref) => true,
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Ref(tc_ref) => Self::opt_cast_from(*tc_ref),
            get_ref if GetRef::can_cast_from(&get_ref) => get_ref.opt_cast_into().map(Self::Get),
            put_ref if PutRef::can_cast_from(&put_ref) => put_ref.opt_cast_into().map(Self::Put),
            post_ref if PostRef::can_cast_from(&post_ref) => {
                post_ref.opt_cast_into().map(Self::Post)
            }
            _ => None,
        }
    }
}

impl<T: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T,) {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>> TryCastFrom<Scalar> for (T1, T2) {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        debug!(
            "cast from {:?} into {}?",
            scalar,
            std::any::type_name::<Self>()
        );

        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

impl<T1: TryCastFrom<Scalar>, T2: TryCastFrom<Scalar>, T3: TryCastFrom<Scalar>> TryCastFrom<Scalar>
    for (T1, T2, T3)
{
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Self> {
        match scalar {
            Scalar::Tuple(tuple) => Self::opt_cast_from(tuple),
            _ => None,
        }
    }
}

/// A [`de::Visitor`] used to deserialize a [`Scalar`].
#[derive(Default)]
pub struct ScalarVisitor {
    value: tc_value::ValueVisitor,
}

impl ScalarVisitor {
    pub async fn visit_map_value<A: de::MapAccess>(
        class: ScalarType,
        access: &mut A,
    ) -> Result<Scalar, A::Error> {
        let scalar = access.next_value::<Scalar>(()).await?;

        if scalar.is_none() {
            // this is a classpath, no need to cast
        } else if let Some(scalar) = scalar.clone().into_type(class) {
            return Ok(scalar);
        } else {
            debug!("cannot cast into {:?} from {:?}", class, scalar);
        }

        let subject = Link::from(class.path()).into();
        Self::visit_subject(subject, scalar)
    }

    pub fn visit_subject<E: de::Error>(subject: Subject, params: Scalar) -> Result<Scalar, E> {
        if params.is_none() {
            let scalar = match subject {
                Subject::Ref(id, path) if path.is_empty() => Scalar::Ref(Box::new(TCRef::Id(id))),
                Subject::Ref(id, path) => Scalar::Ref(Box::new(TCRef::Op(OpRef::Get((
                    Subject::Ref(id, path),
                    Value::default().into(),
                ))))),
                Subject::Link(link) => Scalar::Value(Value::Link(link)),
            };

            Ok(scalar)
        } else {
            OpRefVisitor::visit_ref_value(subject, params)
                .map(TCRef::Op)
                .map(Box::new)
                .map(Scalar::Ref)
        }
    }
}

#[async_trait]
impl de::Visitor for ScalarVisitor {
    type Value = Scalar;

    fn expecting() -> &'static str {
        "a Scalar, e.g. \"foo\" or 123 or {\"$ref: [\"id\", \"$state\"]\"}"
    }

    async fn visit_array_u8<A: de::ArrayAccess<u8>>(
        self,
        array: A,
    ) -> Result<Self::Value, A::Error> {
        self.value.visit_array_u8(array).map_ok(Scalar::Value).await
    }

    fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
        self.value.visit_bool(value).map(Scalar::Value)
    }

    fn visit_i8<E: de::Error>(self, value: i8) -> Result<Self::Value, E> {
        self.value.visit_i8(value).map(Scalar::Value)
    }

    fn visit_i16<E: de::Error>(self, value: i16) -> Result<Self::Value, E> {
        self.value.visit_i16(value).map(Scalar::Value)
    }

    fn visit_i32<E: de::Error>(self, value: i32) -> Result<Self::Value, E> {
        self.value.visit_i32(value).map(Scalar::Value)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        self.value.visit_i64(value).map(Scalar::Value)
    }

    fn visit_u8<E: de::Error>(self, value: u8) -> Result<Self::Value, E> {
        self.value.visit_u8(value).map(Scalar::Value)
    }

    fn visit_u16<E: de::Error>(self, value: u16) -> Result<Self::Value, E> {
        self.value.visit_u16(value).map(Scalar::Value)
    }

    fn visit_u32<E: de::Error>(self, value: u32) -> Result<Self::Value, E> {
        self.value.visit_u32(value).map(Scalar::Value)
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        self.value.visit_u64(value).map(Scalar::Value)
    }

    fn visit_f32<E: de::Error>(self, value: f32) -> Result<Self::Value, E> {
        self.value.visit_f32(value).map(Scalar::Value)
    }

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
        self.value.visit_f64(value).map(Scalar::Value)
    }

    fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
        self.value.visit_string(value).map(Scalar::Value)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        self.value.visit_unit().map(Scalar::Value)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        self.value.visit_none().map(Scalar::Value)
    }

    async fn visit_map<A: de::MapAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let key = if let Some(key) = access.next_key::<String>(()).await? {
            key
        } else {
            return Ok(Scalar::Map(Map::default()));
        };

        if key.starts_with('/') {
            if let Ok(path) = TCPathBuf::from_str(&key) {
                if let Some(class) = ScalarType::from_path(&path) {
                    debug!("decode instance of {:?}", class);
                    return Self::visit_map_value(class, &mut access).await;
                } else {
                    debug!("not a scalar classpath: {}", path);
                }
            }
        }

        if let Ok(subject) = Subject::from_str(&key) {
            let params = access.next_value(()).await?;
            return Self::visit_subject(subject, params);
        }

        let mut map = Map::new();
        let key = Id::from_str(&key).map_err(de::Error::custom)?;
        let value = access.next_value(()).await?;
        map.insert(key, value);

        while let Some(key) = access.next_key(()).await? {
            let value = access.next_value(()).await?;
            map.insert(key, value);
        }

        Ok(Scalar::Map(map.into()))
    }

    async fn visit_seq<A: de::SeqAccess>(self, mut access: A) -> Result<Self::Value, A::Error> {
        let mut items: Vec<Scalar> = if let Some(size) = access.size_hint() {
            Vec::with_capacity(size)
        } else {
            Vec::new()
        };

        while let Some(value) = access.next_element(()).await? {
            items.push(value)
        }

        Ok(Scalar::Tuple(items.into()))
    }
}

#[async_trait]
impl de::FromStream for Scalar {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), d: &mut D) -> Result<Self, D::Error> {
        d.decode_any(ScalarVisitor::default()).await
    }
}

impl<'en> en::ToStream<'en> for Scalar {
    fn to_stream<E: en::Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        match self {
            Scalar::Cluster(ClusterRef(path)) => single_entry(self.class().path(), path, e),
            Scalar::Map(map) => map.to_stream(e),
            Scalar::Op(op_def) => op_def.to_stream(e),
            Scalar::Range(_range) => todo!("Scalar::Range::to_stream"),
            Scalar::Ref(tc_ref) => tc_ref.to_stream(e),
            Scalar::Tuple(tuple) => tuple.to_stream(e),
            Scalar::Value(value) => value.to_stream(e),
        }
    }
}

impl<'en> en::IntoStream<'en> for Scalar {
    fn into_stream<E: en::Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        let classpath = self.class().path();

        match self {
            Scalar::Cluster(ClusterRef(path)) => single_entry(classpath, path, e),
            Scalar::Map(map) => map.into_inner().into_stream(e),
            Scalar::Op(op_def) => op_def.into_stream(e),
            Scalar::Range(_range) => todo!("Scalar::Range::into_stream"),
            Scalar::Ref(tc_ref) => tc_ref.into_stream(e),
            Scalar::Tuple(tuple) => tuple.into_inner().into_stream(e),
            Scalar::Value(value) => value.into_stream(e),
        }
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Scalar::Cluster(cluster) => fmt::Debug::fmt(cluster, f),
            Scalar::Map(map) => fmt::Debug::fmt(map, f),
            Scalar::Op(op) => fmt::Debug::fmt(op, f),
            Scalar::Range(range) => fmt::Debug::fmt(range, f),
            Scalar::Ref(tc_ref) => fmt::Debug::fmt(tc_ref, f),
            Scalar::Tuple(tuple) => fmt::Debug::fmt(tuple, f),
            Scalar::Value(value) => fmt::Debug::fmt(value, f),
        }
    }
}

/// The execution scope of a [`Scalar`], such as an [`OpDef`] or [`TCRef`]
pub struct Scope<'a, State, T> {
    subject: Option<&'a T>,
    data: Map<State>,
}

impl<'a, State, T> Scope<'a, State, T>
where
    State: StateInstance + Refer<State>,
    T: ToState<State> + Instance + Public<State>,
{
    pub fn new<S: Into<State>, I: IntoIterator<Item = (Id, S)>>(
        subject: Option<&'a T>,
        data: I,
    ) -> Self {
        let data = data.into_iter().map(|(id, s)| (id, s.into())).collect();

        debug!("new execution scope: {:?}", data);
        Self { subject, data }
    }

    pub fn with_context<S: Into<State>, I: IntoIterator<Item = (Id, S)>>(
        subject: Option<&'a T>,
        context: Map<State>,
        data: I,
    ) -> Self {
        let data = context
            .into_inner()
            .into_iter()
            .chain(data.into_iter().map(|(id, s)| (id, s.into())))
            .collect();

        debug!("new execution scope: {:?}", data);
        Self { subject, data }
    }

    pub fn into_inner(self) -> Map<State> {
        self.data
    }

    pub fn resolve_id(&self, id: &Id) -> TCResult<State> {
        debug!("resolve ID {}", id);

        let result = if id == &SELF {
            self.subject().map(|subject| subject.to_state())
        } else {
            self.data
                .deref()
                .get(id)
                .cloned()
                .ok_or_else(|| TCError::not_found(format!("state with ID {}", id)))
        };

        match result {
            Ok(state) => {
                debug!("{} resolved to {:?}", id, state);
                Ok(state)
            }
            Err(cause) => {
                warn!(
                    "error resolving {} in context {}: {}",
                    id,
                    self.data.keys().collect::<Tuple<&Id>>(),
                    cause
                );
                Err(cause)
            }
        }
    }

    pub async fn resolve_get(
        &self,
        txn: &State::Txn,
        subject: &Id,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<State> {
        if subject == &SELF {
            let subject = self.subject()?;
            subject.get(txn, path, key).await
        } else if let Some(subject) = self.data.deref().get(subject).cloned() {
            let subject = subject.resolve(self, txn).await?;
            subject.get(txn, path, key).await
        } else {
            Err(TCError::not_found(format!(
                "GET method subject {}",
                subject
            )))
        }
    }

    pub async fn resolve_put(
        &self,
        txn: &State::Txn,
        subject: &Id,
        path: &[PathSegment],
        key: Value,
        value: State,
    ) -> TCResult<()> {
        if subject == &SELF {
            let subject = self.subject()?;
            subject.put(txn, path, key, value).await
        } else if let Some(subject) = self.data.deref().get(subject).cloned() {
            let subject = subject.resolve(self, txn).await?;
            subject.put(txn, path, key, value).await
        } else {
            Err(TCError::not_found(format!(
                "PUT method subject {}",
                subject
            )))
        }
    }

    pub async fn resolve_post(
        &self,
        txn: &State::Txn,
        subject: &Id,
        path: &[PathSegment],
        params: Map<State>,
    ) -> TCResult<State> {
        if subject == &SELF {
            let subject = self.subject()?;
            subject.post(txn, path, params).await
        } else if let Some(subject) = self.data.deref().get(subject).cloned() {
            let subject = subject.resolve(self, txn).await?;
            subject.post(txn, path, params).await
        } else {
            Err(TCError::not_found(format!(
                "POST method subject {}",
                subject
            )))
        }
    }

    pub async fn resolve_delete(
        &self,
        txn: &State::Txn,
        subject: &Id,
        path: &[PathSegment],
        key: Value,
    ) -> TCResult<()> {
        if subject == &SELF {
            let subject = self.subject()?;
            subject.delete(txn, path, key).await
        } else if let Some(subject) = self.data.deref().get(subject).cloned() {
            let subject = subject.resolve(self, txn).await?;
            subject.delete(txn, path, key).await
        } else {
            Err(TCError::not_found(format!(
                "DELETE method subject {}",
                subject
            )))
        }
    }

    fn subject(&self) -> TCResult<&T> {
        if let Some(subject) = self.subject {
            Ok(subject)
        } else {
            Err(bad_request!("{}", ERR_NO_SELF))
        }
    }
}

impl<'a, State, T> Deref for Scope<'a, State, T> {
    type Target = Map<State>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, State, T> DerefMut for Scope<'a, State, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, State, T> fmt::Debug for Scope<'a, State, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ids = self.keys().collect::<Tuple<&Id>>();
        write!(f, "execution scope with IDs {}", ids)
    }
}

fn single_entry<
    'en,
    K: en::IntoStream<'en> + 'en,
    V: en::IntoStream<'en> + 'en,
    E: en::Encoder<'en>,
>(
    key: K,
    value: V,
    encoder: E,
) -> Result<E::Ok, E::Error> {
    use en::EncodeMap;

    let mut map = encoder.encode_map(Some(1))?;
    map.encode_entry(key, value)?;
    map.end()
}
