use std::fmt;

#[derive(Clone)]
pub enum Code {
    BadRequest,
    Internal,
    MethodNotAllowed,
    NotFound,

    #[allow(dead_code)]
    NotImplemented,
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

pub fn bad_request<T: fmt::Display>(message: &str, info: T) -> TCError {
    TCError::of(Code::BadRequest, format!("{}: {}", message, info))
}

pub fn internal(message: &str) -> TCError {
    TCError::of(Code::Internal, message.to_string())
}

pub fn method_not_allowed() -> TCError {
    TCError::of(
        Code::MethodNotAllowed,
        "This resource does not support this request method".to_string(),
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
