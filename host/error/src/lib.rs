//! Provides common error types and associated convenience methods for TinyChain.
//!
//! This crate is a part of TinyChain: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)

use std::convert::Infallible;
use std::{fmt, io};

use destream::en;

/// A result of type `T`, or a [`TCError`]
pub type TCResult<T> = Result<T, TCError>;

struct ErrorData {
    message: String,
    stack: Vec<String>,
}

impl<'en> en::ToStream<'en> for ErrorData {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        if self.stack.is_empty() {
            return en::ToStream::to_stream(&self.message, encoder);
        }

        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(2))?;
        map.encode_entry("message", &self.message)?;
        map.encode_entry("stack", &self.stack)?;
        map.end()
    }
}

impl<'en> en::IntoStream<'en> for ErrorData {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        if self.stack.is_empty() {
            return en::IntoStream::into_stream(self.message, encoder);
        }

        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(2))?;
        map.encode_entry("message", self.message)?;
        map.encode_entry("stack", self.stack)?;
        map.end()
    }
}

impl<T> From<T> for ErrorData
where
    T: fmt::Display,
{
    fn from(message: T) -> Self {
        Self {
            message: message.to_string(),
            stack: vec![],
        }
    }
}

/// The category of a `TCError`.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum ErrorKind {
    BadGateway,
    BadRequest,
    Conflict,
    Forbidden,
    Internal,
    MethodNotAllowed,
    NotFound,
    NotImplemented,
    Timeout,
    Unauthorized,
    Unavailable,
}

impl<'en> en::IntoStream<'en> for ErrorKind {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        format!(
            "/error/{}",
            match self {
                Self::BadGateway => "bad_gateway",
                Self::BadRequest => "bad_request",
                Self::Conflict => "conflict",
                Self::Forbidden => "forbidden",
                Self::Internal => "internal",
                Self::MethodNotAllowed => "method_not_allowed",
                Self::NotFound => "not_found",
                Self::NotImplemented => "not_implemented",
                Self::Timeout => "timeout",
                Self::Unauthorized => "unauthorized",
                Self::Unavailable => "temporarily unavailable",
            }
        )
        .into_stream(encoder)
    }
}

impl fmt::Debug for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            Self::BadGateway => "bad gateway",
            Self::BadRequest => "bad request",
            Self::Conflict => "conflict",
            Self::Forbidden => "forbidden",
            Self::Internal => "internal error",
            Self::MethodNotAllowed => "method not allowed",
            Self::NotFound => "not found",
            Self::NotImplemented => "not implemented",
            Self::Timeout => "request timeout",
            Self::Unauthorized => "unauthorized",
            Self::Unavailable => "temporarily unavailable",
        })
    }
}

/// A general error description.
pub struct TCError {
    kind: ErrorKind,
    data: ErrorData,
}

impl TCError {
    /// Returns a new error with the given code and message.
    pub fn new<I: fmt::Display>(code: ErrorKind, message: I) -> Self {
        #[cfg(debug_assertions)]
        match code {
            ErrorKind::Internal | ErrorKind::MethodNotAllowed | ErrorKind::NotImplemented => {
                panic!("{code}: {message}")
            }
            other => log::warn!("{other}: {message}"),
        }

        Self {
            kind: code,
            data: message.into(),
        }
    }

    /// Reconstruct a [`TCError`] from its [`ErrorType`] and data.
    pub fn with_stack<I, S, SI>(code: ErrorKind, message: I, stack: S) -> Self
    where
        I: fmt::Display,
        SI: fmt::Display,
        S: IntoIterator<Item = SI>,
    {
        let stack = stack.into_iter().map(|msg| msg.to_string()).collect();

        #[cfg(debug_assertions)]
        match code {
            ErrorKind::Internal | ErrorKind::MethodNotAllowed | ErrorKind::NotImplemented => {
                panic!("{code}: {message} (cause: {stack:?})")
            }
            other => log::warn!("{other}: {message} (cause: {stack:?})"),
        }

        Self {
            kind: code,
            data: ErrorData {
                message: message.to_string(),
                stack,
            },
        }
    }

    /// Error to convey an upstream problem
    pub fn bad_gateway<I: fmt::Display>(locator: I) -> Self {
        Self::new(ErrorKind::BadGateway, locator)
    }

    /// Error to indicate that the requested resource is already locked
    pub fn conflict<I: fmt::Display>(locator: I) -> Self {
        Self::new(ErrorKind::Conflict, locator)
    }

    /// Error to indicate that the requested resource exists but does not support the request method
    pub fn method_not_allowed<M: fmt::Debug, S: fmt::Debug, P: fmt::Display>(
        method: M,
        subject: S,
        path: P,
    ) -> Self {
        let message = format!(
            "{:?} endpoint {} does not support {:?}",
            subject, path, method
        );

        Self::new(ErrorKind::MethodNotAllowed, message)
    }

    /// Error to indicate that the requested resource does not exist at the specified location
    pub fn not_found<I: fmt::Display>(locator: I) -> Self {
        Self::new(ErrorKind::NotFound, locator)
    }

    pub fn unexpected<V: fmt::Debug>(value: V, expected: &str) -> Self {
        Self::new(
            ErrorKind::BadRequest,
            format!("invalid value {value:?}: expected {expected}"),
        )
    }

    pub fn unsupported<I: fmt::Display>(info: I) -> Self {
        Self::new(ErrorKind::BadRequest, info.to_string())
    }

    /// The [`ErrorKind`] of this error
    pub fn code(&self) -> ErrorKind {
        self.kind
    }

    /// The error message of this error
    pub fn message(&'_ self) -> &'_ str {
        &self.data.message
    }

    /// Construct a new error with the given `cause`
    pub fn consume<I: fmt::Debug>(mut self, cause: I) -> Self {
        self.data.stack.push(format!("{:?}", cause));
        self
    }
}

impl std::error::Error for TCError {}

impl From<pathlink::ParseError> for TCError {
    fn from(err: pathlink::ParseError) -> Self {
        Self::new(ErrorKind::BadRequest, err)
    }
}

impl From<ha_ndarray::Error> for TCError {
    fn from(err: ha_ndarray::Error) -> Self {
        Self::new(ErrorKind::Internal, err)
    }
}

impl From<txn_lock::Error> for TCError {
    fn from(err: txn_lock::Error) -> Self {
        Self::new(ErrorKind::Conflict, err)
    }
}

impl From<txfs::Error> for TCError {
    fn from(cause: txfs::Error) -> Self {
        match cause.into_inner() {
            (txfs::ErrorKind::NotFound, msg) => Self::not_found(msg),
            (txfs::ErrorKind::Conflict, msg) => Self::conflict(msg),
            (txfs::ErrorKind::IO, msg) => Self::bad_gateway(msg),
        }
    }
}

#[cfg(debug_assertions)]
impl From<io::Error> for TCError {
    fn from(cause: io::Error) -> Self {
        panic!("IO error: {cause}");
    }
}

#[cfg(not(debug_assertions))]
impl From<io::Error> for TCError {
    fn from(cause: io::Error) -> Self {
        match cause.kind() {
            io::ErrorKind::AlreadyExists => bad_request!(
                "tried to create a filesystem entry that already exists: {}",
                cause
            ),
            io::ErrorKind::InvalidInput => bad_request!("{}", cause),
            io::ErrorKind::NotFound => TCError::not_found(cause),
            io::ErrorKind::PermissionDenied => {
                bad_gateway!("host filesystem permission denied").consume(cause)
            }
            io::ErrorKind::WouldBlock => {
                conflict!("synchronous filesystem access failed").consume(cause)
            }
            kind => internal!("host filesystem error: {:?}", kind).consume(cause),
        }
    }
}

impl From<Infallible> for TCError {
    fn from(_: Infallible) -> Self {
        internal!("an unanticipated error occurred--please file a bug report")
    }
}

impl<'en> en::ToStream<'en> for TCError {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.kind, &self.data)?;
        map.end()
    }
}

impl<'en> en::IntoStream<'en> for TCError {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::EncodeMap;
        let mut map = encoder.encode_map(Some(1))?;
        map.encode_entry(self.kind, self.data)?;
        map.end()
    }
}

impl fmt::Debug for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for TCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.data.message)
    }
}

/// Error to convey an upstream problem
#[macro_export]
macro_rules! bad_gateway {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::BadGateway, format!($($t)*))
    }}
}

/// Error to indicate that the request is badly-constructed or nonsensical
#[macro_export]
macro_rules! bad_request {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::BadRequest, format!($($t)*))
    }}
}

/// Error to indicate that the request cannot be fulfilled due to a conflict with another request.
#[macro_export]
macro_rules! conflict {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Conflict, format!($($t)*))
    }}
}

/// Error to indicate that the requestor's credentials do not authorize the request to be fulfilled
#[macro_export]
macro_rules! forbidden {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Unavailable, format!($($t)*))
    }}
}

/// Error to indicate that a required feature is not yet implemented.
#[macro_export]
macro_rules! not_implemented {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::NotImplemented, format!($($t)*))
    }}
}

/// Error to indicate that the request failed to complete in the allotted time.
#[macro_export]
macro_rules! timeout {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Timeout, format!($($t)*))
    }}
}

/// A truly unexpected error, for which no handling behavior can be defined
#[macro_export]
macro_rules! internal {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Internal, format!($($t)*))
    }}
}

/// Error to indicate that the user's credentials are missing or nonsensical.
#[macro_export]
macro_rules! unauthorized {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Unauthorized, format!($($t)*))
    }}
}

/// Error to indicate that this host is currently overloaded
#[macro_export]
macro_rules! unavailable {
    ($($t:tt)*) => {{
        $crate::TCError::new($crate::ErrorKind::Unavailable, format!($($t)*))
    }}
}
