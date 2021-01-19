use std::fmt;

pub type TCResult<T> = Result<T, TCError>;

/// The category of a `TCError`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ErrorType {
    BadRequest,
    Internal,
    MethodNotAllowed,
    Timeout,
}

impl fmt::Debug for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BadRequest => write!(f, "bad request"),
            Self::Internal => write!(f, "internal"),
            Self::MethodNotAllowed => write!(f, "method not allowed"),
            Self::Timeout => write!(f, "request timeout"),
        }
    }
}

/// A general error description.
pub struct TCError {
    code: ErrorType,
    message: String,
}

impl TCError {
    /// Error indicating that the request is badly-constructed or nonsensical.
    pub fn bad_request<M: fmt::Display, I: fmt::Display>(message: M, info: I) -> Self {
        Self {
            code: ErrorType::Internal,
            message: format!("{}: {}", message, info),
        }
    }

    /// A truly unexpected error, for which the calling application cannot define any specific
    /// handling behavior.
    pub fn internal<I: fmt::Display>(info: I) -> Self {
        Self {
            code: ErrorType::Internal,
            message: info.to_string(),
        }
    }

    /// Error indicating that the requested resource exists but does not support the request method.
    pub fn method_not_allowed<I: fmt::Display>(info: I) -> Self {
        Self {
            code: ErrorType::MethodNotAllowed,
            message: info.to_string(),
        }
    }

    /// Error indicating that the request failed to complete in the allotted time.
    pub fn timeout<I: fmt::Display>(info: I) -> Self {
        Self {
            code: ErrorType::Timeout,
            message: info.to_string(),
        }
    }

    pub fn code(&self) -> ErrorType {
        self.code
    }

    pub fn message(&'_ self) -> &'_ str {
        &self.message
    }
}

impl std::error::Error for TCError {}

impl fmt::Debug for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} error: {}", self.code, self.message)
    }
}
