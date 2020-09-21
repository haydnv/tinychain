use std::convert::Infallible;
use std::fmt;

use crate::value::{label, TCPath};

pub type TCResult<T> = Result<T, TCError>;

#[derive(Clone)]
pub enum Code {
    // "No problem"
    Ok,

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

impl From<u16> for Code {
    fn from(code: u16) -> Code {
        use Code::*;

        match code {
            200 => Ok,
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

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Code::Ok => write!(f, "Ok"),
            Code::BadRequest => write!(f, "Bad request"),
            Code::Conflict => write!(f, "Conflict"),
            Code::Forbidden => write!(f, "Forbidden"),
            Code::Internal => write!(f, "Internal server error"),
            Code::MethodNotAllowed => write!(f, "Method not allowed"),
            Code::NotFound => write!(f, "Not found"),
            Code::NotImplemented => write!(f, "Not implemented"),
            Code::TooLarge => write!(f, "Request too large"),
            Code::Transport => write!(f, "Transport protocol error"),
            Code::Unauthorized => write!(f, "Unauthorized"),
            Code::Unknown => write!(f, "Unrecognized error code"),
        }
    }
}

#[derive(Clone)]
pub struct TCError {
    reason: Code,
    message: String,
}

impl TCError {
    pub fn of(reason: Code, message: String) -> TCError {
        TCError { reason, message }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn reason(&self) -> &Code {
        &self.reason
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

pub fn get(path: &TCPath, msg: String) -> TCError {
    let prefix = TCPath::from(vec![label("sbin").into(), label("error").into()]);
    let suffix = match path.from_path(&prefix) {
        Ok(suffix) => suffix,
        Err(cause) => return cause,
    };

    let (code, msg) = match suffix[0].as_str() {
        "bad_request" => (Code::BadRequest, msg),
        "conflict" => (Code::Conflict, msg),
        "forbidden" => (Code::Forbidden, msg),
        "internal" => (Code::Internal, msg),
        "method_not_allowed" => (Code::MethodNotAllowed, msg),
        "not_found" => (Code::NotFound, msg),
        "not_implemented" => (Code::NotImplemented, msg),
        "too_large" => (Code::TooLarge, msg),
        "transport" => (Code::Transport, msg),
        "unauthorized" => (Code::Unauthorized, msg),
        "unknown" => (Code::Unknown, msg),
        _ => (Code::NotFound, suffix.to_string()),
    };

    TCError::of(code, msg)
}

pub fn bad_request<T: fmt::Display>(message: &str, info: T) -> TCError {
    println!("error! {}: {}", message, info);
    TCError::of(Code::BadRequest, format!("{}: {}", message, info))
}

pub fn conflict() -> TCError {
    TCError::of(
        Code::Conflict,
        "Transaction failed due to a concurrent access conflict".to_string(),
    )
}

pub fn forbidden<T: fmt::Display>(message: &str, info: T) -> TCError {
    TCError::of(Code::Forbidden, format!("{}: {}", message, info))
}

pub fn internal<T: fmt::Display>(cause: T) -> TCError {
    TCError::of(Code::Internal, format!("{}", cause))
}

pub fn method_not_allowed<T: fmt::Display>(id: T) -> TCError {
    TCError::of(
        Code::MethodNotAllowed,
        format!("This resource does not support this request method: {}", id),
    )
}

pub fn not_found<T: fmt::Display>(id: T) -> TCError {
    TCError::of(
        Code::NotFound,
        format!("The requested resource could not be found: {}", id),
    )
}

#[allow(dead_code)]
pub fn not_implemented<I: fmt::Display>(feature: I) -> TCError {
    TCError::of(
        Code::NotImplemented,
        format!("This feature is not yet implemented: {}", feature),
    )
}

pub fn unsupported<I: fmt::Display>(hint: I) -> TCError {
    TCError::of(Code::BadRequest, hint.to_string())
}

pub fn too_large(max_size: usize) -> TCError {
    TCError::of(
        Code::TooLarge,
        format!(
            "Payload exceeded the maximum allowed size of {} bytes",
            max_size,
        ),
    )
}

pub fn transport<I: fmt::Display>(info: I) -> TCError {
    TCError::of(Code::Transport, info.to_string())
}

pub fn unauthorized(message: &str) -> TCError {
    TCError::of(Code::Unauthorized, message.into())
}
