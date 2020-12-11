use std::convert::Infallible;
use std::fmt;

use crate::class::{Class, Instance, NativeClass, TCType};
use crate::scalar::value::{label, Link, PathSegment, TCPath, TCPathBuf};

pub type TCResult<T> = Result<T, TCError>;

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ErrorType {
    // "I know that what you're asking for doesn't make sense"
    BadRequest,

    // "Another caller has this resource reserved, so your request cannot be fulfilled"
    Conflict,

    // "I know who you are and you're not allowed to do this!"
    Forbidden,

    // "Something that really should have worked didn't work--you should file a bug report"
    Internal,

    // "This resource exists but it doesn't support the request method you used"
    MethodNotAllowed,

    // "I don't know what this is--maybe you're looking in the wrong place?"
    NotFound,

    // "This is marked for implementation in the future"
    NotImplemented,

    // "The payload itself is dangerously large"
    TooLarge,

    // "There was an error at the transport protocol layer while handling your request"
    Transport,

    // "This resource requires authorization but your credentials are absent or nonsensical"
    Unauthorized,

    // "A downstream dependency responded with an unrecognized status code"
    Unknown,
}

impl ErrorType {
    pub fn get<I: fmt::Display>(self, message: I) -> TCError {
        TCError {
            reason: self,
            message: message.to_string(),
        }
    }
}

impl Class for ErrorType {
    type Instance = TCError;
}

impl NativeClass for ErrorType {
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            let err = match suffix[0].as_str() {
                "bad_request" => Self::BadRequest,
                "unauthorized" => Self::Unauthorized,
                "forbidden" => Self::Forbidden,
                "not_found" => Self::NotFound,
                "method_not_allowed" => Self::MethodNotAllowed,
                "conflict" => Self::Conflict,
                "too_large" => Self::TooLarge,
                "transport" => Self::Transport,
                "internal" => Self::Internal,
                "not_implemented" => Self::NotImplemented,
                "unknown" => Self::Unknown,
                other => return Err(not_found(other)),
            };

            Ok(err)
        } else {
            Err(path_not_found(path))
        }
    }

    fn prefix() -> TCPathBuf {
        TCType::prefix().append(label("error"))
    }
}

impl From<u16> for ErrorType {
    fn from(code: u16) -> ErrorType {
        use ErrorType::*;

        match code {
            400 => BadRequest,
            401 => Unauthorized,
            403 => Forbidden,
            404 => NotFound,
            405 => MethodNotAllowed,
            409 => Conflict,
            413 => TooLarge,
            499 => Transport,
            500 => Internal,
            501 => NotImplemented,
            _ => Unknown,
        }
    }
}

impl From<ErrorType> for Link {
    fn from(et: ErrorType) -> Link {
        use ErrorType::*;

        let label = match et {
            BadRequest => label("bad_request"),
            Unauthorized => label("unauthorized"),
            Forbidden => label("forbidden"),
            NotFound => label("not_found"),
            MethodNotAllowed => label("method_not_allowed"),
            Conflict => label("conflict"),
            TooLarge => label("too_large"),
            Transport => label("transport"),
            Internal => label("internal"),
            NotImplemented => label("not_implemented"),
            Unknown => label("unknown"),
        };

        ErrorType::prefix().append(label).into()
    }
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ErrorType::BadRequest => write!(f, "Bad request"),
            ErrorType::Conflict => write!(f, "Conflict"),
            ErrorType::Forbidden => write!(f, "Forbidden"),
            ErrorType::Internal => write!(f, "Internal server error"),
            ErrorType::MethodNotAllowed => write!(f, "Method not allowed"),
            ErrorType::NotFound => write!(f, "Not found"),
            ErrorType::NotImplemented => write!(f, "Not implemented"),
            ErrorType::TooLarge => write!(f, "Request too large"),
            ErrorType::Transport => write!(f, "Transport protocol error"),
            ErrorType::Unauthorized => write!(f, "Unauthorized"),
            ErrorType::Unknown => write!(f, "Unrecognized error code"),
        }
    }
}

#[derive(Clone)]
pub struct TCError {
    reason: ErrorType,
    message: String,
}

impl TCError {
    pub fn of(reason: ErrorType, message: String) -> TCError {
        TCError { reason, message }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn reason(&self) -> &ErrorType {
        &self.reason
    }
}

impl Instance for TCError {
    type Class = ErrorType;

    fn class(&self) -> Self::Class {
        self.reason
    }
}

impl From<Box<bincode::ErrorKind>> for TCError {
    fn from(e: Box<bincode::ErrorKind>) -> TCError {
        bad_request("Serialization error", e)
    }
}

impl From<Infallible> for TCError {
    fn from(e: Infallible) -> TCError {
        internal(format!("Internal system error: {}", e))
    }
}

impl From<hyper::Error> for TCError {
    fn from(e: hyper::Error) -> TCError {
        transport(format!("HTTP interface error: {}", e))
    }
}

impl From<serde_json::error::Error> for TCError {
    fn from(e: serde_json::error::Error) -> TCError {
        bad_request("Serialization error", e)
    }
}

impl std::error::Error for TCError {}

impl fmt::Debug for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl fmt::Display for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.reason, self.message)
    }
}

pub fn bad_request<I: fmt::Display, T: fmt::Display>(message: I, info: T) -> TCError {
    TCError::of(ErrorType::BadRequest, format!("{}: {}", message, info))
}

pub fn conflict() -> TCError {
    TCError::of(
        ErrorType::Conflict,
        "Transaction failed due to a concurrent access conflict".to_string(),
    )
}

pub fn forbidden<I: fmt::Display, T: fmt::Display>(message: I, info: T) -> TCError {
    TCError::of(ErrorType::Forbidden, format!("{}: {}", message, info))
}

pub fn internal<T: fmt::Display>(cause: T) -> TCError {
    TCError::of(ErrorType::Internal, format!("{}", cause))
}

pub fn method_not_allowed<T: fmt::Display>(id: T) -> TCError {
    TCError::of(
        ErrorType::MethodNotAllowed,
        format!("This resource does not support this request method: {}", id),
    )
}

pub fn not_found<T: fmt::Display>(id: T) -> TCError {
    TCError::of(
        ErrorType::NotFound,
        format!("The requested resource could not be found: {}", id),
    )
}

pub fn path_not_found(path: &[PathSegment]) -> TCError {
    not_found(TCPath::from(path))
}

#[allow(dead_code)]
pub fn not_implemented<I: fmt::Display>(feature: I) -> TCError {
    TCError::of(
        ErrorType::NotImplemented,
        format!("This feature is not yet implemented: {}", feature),
    )
}

pub fn unsupported<I: fmt::Display>(hint: I) -> TCError {
    TCError::of(ErrorType::BadRequest, hint.to_string())
}

pub fn too_large(max_size: usize) -> TCError {
    TCError::of(
        ErrorType::TooLarge,
        format!(
            "Payload exceeded the maximum allowed size of {} bytes",
            max_size,
        ),
    )
}

pub fn transport<I: fmt::Display>(info: I) -> TCError {
    TCError::of(ErrorType::Transport, info.to_string())
}

pub fn unauthorized(message: &str) -> TCError {
    TCError::of(ErrorType::Unauthorized, message.into())
}
