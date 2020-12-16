use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::TryFutureExt;
use serde::ser::{Serialize, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCResult, TCType};
use crate::error;
use crate::handler::*;
use crate::request::Request;
use crate::transaction::Txn;

use super::reference::Refer;
use super::value::{label, Id, Link, PathSegment, TCPathBuf};
use super::{MethodType, Scalar, ScalarInstance, ScalarType, TryCastFrom, TryCastInto};

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct MapType;

impl Class for MapType {
    type Instance = Map;
}

impl NativeClass for MapType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        if path == Self::prefix().as_slice() {
            Ok(Self)
        } else {
            Err(error::path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        ScalarType::prefix().append(label("map"))
    }
}

impl From<MapType> for Link {
    fn from(_ot: MapType) -> Link {
        MapType::prefix().into()
    }
}

impl From<MapType> for TCType {
    fn from(_: MapType) -> TCType {
        ScalarType::Map.into()
    }
}

impl fmt::Display for MapType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Scalar Map")
    }
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct Map(HashMap<Id, Scalar>);

impl Map {
    pub fn into_inner(self) -> HashMap<Id, Scalar> {
        self.0
    }
}

impl Instance for Map {
    type Class = MapType;

    fn class(&self) -> Self::Class {
        MapType
    }
}

#[async_trait]
impl Refer for Map {
    fn requires(&self, deps: &mut HashSet<Id>) {
        for tc_ref in self.values() {
            tc_ref.requires(deps);
        }
    }

    async fn resolve(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State>,
    ) -> TCResult<State> {
        let mut map = HashMap::<Id, State>::new();
        let mut pending = FuturesUnordered::from_iter(self.0.into_iter().map(|(id, scalar)| {
            scalar
                .resolve(request, txn, context)
                .map_ok(|state| (id, state))
        }));

        while let Some(result) = pending.next().await {
            let (id, state) = result?;
            map.insert(id, state);
        }

        if Map::can_cast_from(&map) {
            Ok(State::Scalar(Scalar::Map(map.opt_cast_into().unwrap())))
        } else {
            Ok(State::Map(Box::new(map)))
        }
    }
}

impl Route for Map {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if path.is_empty() {
            return None;
        }

        if let Some(scalar) = self.0.get(&path[0]) {
            match scalar {
                Scalar::Op(op_def) if path.len() == 1 => {
                    Some(op_def.handler(Some(self.clone().into())))
                }
                scalar => scalar.route(method, &path[1..]),
            }
        } else {
            None
        }
    }
}

impl Deref for Map {
    type Target = HashMap<Id, Scalar>;

    fn deref(&'_ self) -> &'_ HashMap<Id, Scalar> {
        &self.0
    }
}

impl DerefMut for Map {
    fn deref_mut(&'_ mut self) -> &'_ mut HashMap<Id, Scalar> {
        &mut self.0
    }
}

impl<T: Into<Scalar>> FromIterator<(Id, T)> for Map {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> Self {
        let mut map = HashMap::new();

        for (id, attr) in iter {
            let scalar = attr.into();
            map.insert(id, scalar);
        }

        Map(map)
    }
}

impl From<HashMap<Id, Scalar>> for Map {
    fn from(map: HashMap<Id, Scalar>) -> Map {
        Map(map)
    }
}

impl TryCastFrom<Scalar> for Map {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Map(_) => true,
            other => other.matches::<Vec<(Id, Scalar)>>(),
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Map> {
        match scalar {
            Scalar::Map(map) => Some(map),
            other if other.matches::<Vec<(Id, Scalar)>>() => {
                let data: Vec<(Id, Scalar)> = other.opt_cast_into().unwrap();
                Some(Map::from_iter(data))
            }
            _ => None,
        }
    }
}

impl TryCastFrom<HashMap<Id, State>> for Map {
    fn can_cast_from(map: &HashMap<Id, State>) -> bool {
        map.values().all(State::is_scalar)
    }

    fn opt_cast_from(map: HashMap<Id, State>) -> Option<Map> {
        let mut cast = HashMap::<Id, Scalar>::new();
        for (id, state) in map.into_iter() {
            if let State::Scalar(scalar) = state {
                cast.insert(id, scalar);
            } else {
                return None;
            }
        }

        Some(Map(cast))
    }
}

impl TryCastFrom<State> for Map {
    fn can_cast_from(state: &State) -> bool {
        if let State::Scalar(scalar) = state {
            Map::can_cast_from(scalar)
        } else if let State::Map(map) = state {
            Map::can_cast_from(&**map)
        } else {
            false
        }
    }

    fn opt_cast_from(state: State) -> Option<Map> {
        if let State::Scalar(scalar) = state {
            Map::opt_cast_from(scalar)
        } else if let State::Map(map) = state {
            Map::opt_cast_from(*map)
        } else {
            None
        }
    }
}

impl From<Map> for HashMap<Id, Scalar> {
    fn from(map: Map) -> HashMap<Id, Scalar> {
        map.0
    }
}

impl From<Map> for State {
    fn from(map: Map) -> State {
        State::Scalar(Scalar::Map(map))
    }
}

impl Serialize for Map {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(s)
    }
}

impl fmt::Display for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.0
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
