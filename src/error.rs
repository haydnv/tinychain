use std::convert;
use std::fmt;

#[derive(Clone)]
pub enum Code {
    // "I know that what you're asking for doesn't make sense"
    BadRequest,

    // "Something that really should have worked didn't work--you should file a bug report"
    #[allow(dead_code)]
    Internal,

    // "I found this but it doesn't support the request method you used (e.g. GET, PUT, POST...)"
    MethodNotAllowed,

    // "I don't know what this is--maybe you're looking in the wrong place?"
    NotFound,

    // "This is marked for implementation in the future"
    #[allow(dead_code)]
    NotImplemented,
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Code::BadRequest => write!(f, "Bad request"),
            Code::Internal => write!(f, "Internal server error"),
            Code::MethodNotAllowed => write!(f, "Method not allowed"),
            Code::NotFound => write!(f, "Not found"),
            Code::NotImplemented => write!(f, "Not implemented"),
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

impl convert::From<serde_json::error::Error> for TCError {
    fn from(e: serde_json::error::Error) -> TCError {
        bad_request("Serialization error", e)
    }
}

impl std::error::Error for TCError {}

impl fmt::Debug for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{ file: {}, line: {} }}", file!(), line!())
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

pub fn internal(message: &str) -> TCError {
    TCError::of(Code::Internal, message.to_string())
}

pub fn method_not_allowed<T: fmt::Display>(context: T) -> TCError {
    TCError::of(
        Code::MethodNotAllowed,
        format!("{} does not support this request method", context),
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
        "This functionality is not yet implemented".to_string(),
    )
}
