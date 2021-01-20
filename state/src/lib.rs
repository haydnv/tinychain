use generic::*;

pub use scalar::*;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum StateType {
    Map,
    Scalar(ScalarType),
    Tuple,
}

impl Class for StateType {
    type Instance = State;
}

impl NativeClass for StateType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.is_empty() {
            None
        } else if &path[0] == "state" {
            if path.len() == 2 {
                match path[1].as_str() {
                    "map" => Some(Self::Map),
                    "tuple" => Some(Self::Tuple),
                    _ => None,
                }
            } else if path.len() > 2 && &path[1] == "scalar" {
                ScalarType::from_path(path).map(Self::Scalar)
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        match self {
            Self::Map => path_label(&["state", "map"]).into(),
            Self::Scalar(st) => st.path(),
            Self::Tuple => path_label(&["state", "tuple"]).into(),
        }
    }
}

#[derive(Clone)]
pub enum State {
    Map(Map<Self>),
    Scalar(Scalar),
    Tuple(Tuple<Self>),
}

impl Instance for State {
    type Class = StateType;

    fn class(&self) -> StateType {
        match self {
            Self::Map(_) => StateType::Map,
            Self::Scalar(scalar) => StateType::Scalar(scalar.class()),
            Self::Tuple(_) => StateType::Tuple,
        }
    }
}
