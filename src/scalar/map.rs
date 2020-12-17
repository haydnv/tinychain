use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter::FromIterator;
use std::ops::Deref;

use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use futures::TryFutureExt;
use serde::ser::{Serialize, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCType};
use crate::error;
use crate::general::{Map, TCResult};
use crate::handler::*;
use crate::request::Request;
use crate::transaction::Txn;

use super::reference::Refer;
use super::value::{label, Id, Link, PathSegment, TCPathBuf};
use super::{MethodType, Scalar, ScalarInstance, ScalarType, TryCastFrom, TryCastInto};

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct MapType;

impl Class for MapType {
    type Instance = Map<Scalar>;
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

impl Instance for Map<Scalar> {
    type Class = MapType;

    fn class(&self) -> Self::Class {
        MapType
    }
}

#[async_trait]
impl Refer for Map<Scalar> {
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
        let mut pending =
            FuturesUnordered::from_iter(self.into_inner().into_iter().map(|(id, scalar)| {
                scalar
                    .resolve(request, txn, context)
                    .map_ok(|state| (id, state))
            }));

        while let Some(result) = pending.next().await {
            let (id, state) = result?;
            map.insert(id, state);
        }

        let map = Map::from(map);
        if Map::<Scalar>::can_cast_from(&map) {
            Ok(State::Scalar(Scalar::Map(map.opt_cast_into().unwrap())))
        } else {
            Ok(State::Map(map.into()))
        }
    }
}

impl Route for Map<Scalar> {
    fn route(&'_ self, method: MethodType, path: &[PathSegment]) -> Option<Box<dyn Handler + '_>> {
        if path.is_empty() {
            return None;
        }

        if let Some(scalar) = self.deref().get(&path[0]) {
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

impl TryCastFrom<Scalar> for Map<Scalar> {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Map(_) => true,
            other => other.matches::<Vec<(Id, Scalar)>>(),
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<Map<Scalar>> {
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

impl TryCastFrom<Map<State>> for Map<Scalar> {
    fn can_cast_from(map: &Map<State>) -> bool {
        map.values().all(State::is_scalar)
    }

    fn opt_cast_from(map: Map<State>) -> Option<Map<Scalar>> {
        let mut cast = HashMap::<Id, Scalar>::new();
        for (id, state) in map.into_inner().into_iter() {
            if let State::Scalar(scalar) = state {
                cast.insert(id, scalar);
            } else {
                return None;
            }
        }

        Some(Map::from(cast))
    }
}

impl TryCastFrom<State> for Map<Scalar> {
    fn can_cast_from(state: &State) -> bool {
        if let State::Scalar(scalar) = state {
            Map::<Scalar>::can_cast_from(scalar)
        } else if let State::Map(map) = state {
            Map::<Scalar>::can_cast_from(map)
        } else {
            false
        }
    }

    fn opt_cast_from(state: State) -> Option<Map<Scalar>> {
        if let State::Scalar(scalar) = state {
            Map::<Scalar>::opt_cast_from(scalar)
        } else if let State::Map(map) = state {
            Map::<Scalar>::opt_cast_from(map)
        } else {
            None
        }
    }
}

impl From<Map<Scalar>> for State {
    fn from(map: Map<Scalar>) -> State {
        State::Scalar(Scalar::Map(map))
    }
}

impl Serialize for Map<Scalar> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.deref().serialize(s)
    }
}
