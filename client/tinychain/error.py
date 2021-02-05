from .util import *


class TinychainError(Exception):
    __ref__ = URI("/error")

    def __init__(self, message):
        self.message = message

    def __json__(self):
        return {str(uri(self)): self.message}


class BadRequest(TinychainError):
    __ref__ = uri(TinychainError) + "/bad_request"


class Forbidden(TinychainError):
    __ref__ = uri(TinychainError) + "/forbidden"


class MethodNotAllowed(TinychainError):
    __ref__ = uri(TinychainError) + "/method_not_allowed"


class NotFound(TinychainError):
    __ref__ = uri(TinychainError) + "/not_found"


class NotImplemented(TinychainError):
    __ref__ = uri(TinychainError) + "/not_implemented"


class Unauthorized(TinychainError):
    __ref__ = uri(TinychainError) + "/unauthorized"


class UnknownError(TinychainError):
    __ref__ = uri(TinychainError) + "/unknown"

