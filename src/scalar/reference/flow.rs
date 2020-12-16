use std::fmt;

use async_trait::async_trait;
use futures::future::try_join_all;
use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::class::{Class, Instance, NativeClass, State, TCResult, TCType};
use crate::error;
use crate::scalar::{
    label, Id, Link, PathSegment, Scalar, ScalarClass, ScalarInstance, TCPathBuf, TryCastFrom,
    TryCastInto, Value,
};

use super::{RefType, Refer, TCRef};
use crate::request::Request;
use crate::transaction::Txn;
use std::collections::hash_map::RandomState;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum FlowControlType {
    After,
    If,
}

impl Class for FlowControlType {
    type Instance = FlowControl;
}

impl NativeClass for FlowControlType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "after" => Ok(Self::After),
                "if" => Ok(Self::If),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        RefType::prefix().append(label("flow"))
    }
}

impl ScalarClass for FlowControlType {
    type Instance = FlowControl;

    fn try_cast<S>(&self, scalar: S) -> TCResult<FlowControl>
    where
        Scalar: From<S>,
    {
        FlowControl::try_cast_from(Scalar::from(scalar), |s| {
            error::bad_request("Cannot cast into FlowControl from", s)
        })
    }
}

impl From<FlowControlType> for Link {
    fn from(fct: FlowControlType) -> Link {
        use FlowControlType as FCT;
        let suffix = match fct {
            FCT::After => label("after"),
            FCT::If => label("if"),
        };

        FCT::prefix().append(suffix).into()
    }
}

impl From<FlowControlType> for TCType {
    fn from(fct: FlowControlType) -> TCType {
        RefType::Flow(fct).into()
    }
}

impl fmt::Display for FlowControlType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "type: control flow - {}",
            match self {
                Self::After => "after",
                Self::If => "if",
            }
        )
    }
}

type After = (Vec<TCRef>, TCRef);
type If = (TCRef, Scalar, Scalar);

#[derive(Clone, Eq, PartialEq)]
pub enum FlowControl {
    After(After),
    If(If),
}

impl Instance for FlowControl {
    type Class = FlowControlType;

    fn class(&self) -> FlowControlType {
        match self {
            Self::After(_) => FlowControlType::After,
            Self::If(_) => FlowControlType::If,
        }
    }
}

impl ScalarInstance for FlowControl {
    type Class = FlowControlType;
}

#[async_trait]
impl Refer for FlowControl {
    fn requires(&self, deps: &mut HashSet<Id, RandomState>) {
        match self {
            FlowControl::After((when, _)) => {
                for tc_ref in when {
                    tc_ref.requires(deps);
                }
            }
            FlowControl::If((cond, _, _)) => {
                cond.requires(deps);
            }
        }
    }

    async fn resolve(
        self,
        request: &Request,
        txn: &Txn,
        context: &HashMap<Id, State, RandomState>,
    ) -> TCResult<State> {
        match self {
            FlowControl::After((when, then)) => {
                try_join_all(
                    when.into_iter()
                        .map(|tc_ref| tc_ref.resolve(request, txn, context)),
                )
                .await?;

                Ok(State::Scalar(Scalar::Ref(Box::new(then))))
            }
            FlowControl::If((cond, then, or_else)) => {
                const ERR_NOT_BOOLEAN: &str = "Expected a boolean condition but found";

                let cond = cond.resolve(request, txn, context).await?;
                let cond: bool = if let State::Scalar(Scalar::Value(Value::Number(cond))) = cond {
                    cond.try_cast_into(|v| error::bad_request(ERR_NOT_BOOLEAN, v))
                } else {
                    Err(error::bad_request(ERR_NOT_BOOLEAN, cond))
                }?;

                if cond {
                    Ok(State::Scalar(then))
                } else {
                    Ok(State::Scalar(or_else))
                }
            }
        }
    }
}

impl TryCastFrom<Scalar> for FlowControl {
    fn can_cast_from(s: &Scalar) -> bool {
        s.matches::<If>() || s.matches::<After>()
    }

    fn opt_cast_from(s: Scalar) -> Option<Self> {
        if s.matches::<If>() {
            s.opt_cast_into().map(Self::If)
        } else if s.matches::<After>() {
            s.opt_cast_into().map(Self::After)
        } else {
            None
        }
    }
}

impl Serialize for FlowControl {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let class = Link::from(self.class()).to_string();
        let mut map = s.serialize_map(Some(1))?;

        match self {
            Self::After((when, then)) => {
                map.serialize_entry(&class, &(when, then))?;
            }
            Self::If((cond, then, or_else)) => {
                map.serialize_entry(&class, &(cond, then, or_else))?;
            }
        }

        map.end()
    }
}

impl fmt::Display for FlowControl {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::After((when, then)) => {
                let when = when
                    .iter()
                    .map(|tc_ref| tc_ref.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");

                write!(f, "After [{}] then {}", when, then)
            }
            Self::If((cond, then, or_else)) => {
                write!(f, "If ({}) then {} else {}", cond, then, or_else)
            }
        }
    }
}
