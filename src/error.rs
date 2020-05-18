use std::convert::Infallible;
use std::fmt;

#[derive(Clone)]
pub enum Code {
    // "I know that what you're asking for doesn't make sense"
    BadRequest,

    // "I know who you are and you're not allowed to do this!"
    Forbidden,

    // "Something that really should have worked didn't work--you should file a bug report"
    #[allow(dead_code)]
    Internal,

    // "This resource exists but it doesn't support the request method you used"
    MethodNotAllowed,

    // "I don't know what this is--maybe you're looking in the wrong place?"
    NotFound,

    // "This is marked for implementation in the future"
    #[allow(dead_code)]
    NotImplemented,

    // "This resource requires authorization but your credentials are absent or nonsensical"
    Unauthorized,
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Code::BadRequest => write!(f, "Bad request"),
            Code::Forbidden => write!(f, "Forbidden"),
            Code::Internal => write!(f, "Internal server error"),
            Code::MethodNotAllowed => write!(f, "Method not allowed"),
            Code::NotFound => write!(f, "Not found"),
            Code::NotImplemented => write!(f, "Not implemented"),
            Code::Unauthorized => write!(f, "Unauthorized"),
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

impl From<Infallible> for TCError {
    fn from(e: Infallible) -> TCError {
        internal(format!("Internal system error: {}", e))
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

pub fn bad_request<T: fmt::Display>(message: &str, info: T) -> TCError {
    TCError::of(Code::BadRequest, format!("{}: {}", message, info))
}

pub fn forbidden(message: &str) -> TCError {
    TCError::of(Code::Forbidden, message.into())
}

pub fn internal<T: fmt::Display>(cause: T) -> TCError {
    TCError::of(Code::Internal, format!("{}", cause))
}

pub fn method_not_allowed<T: fmt::Display>(id: T) -> TCError {
    TCError::of(
        Code::Internal,
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
pub fn not_implemented() -> TCError {
    TCError::of(
        Code::NotImplemented,
        "This functionality is not yet implemented".into(),
    )
}

pub fn unauthorized(message: &str) -> TCError {
    TCError::of(Code::Unauthorized, message.into())
}
