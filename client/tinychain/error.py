from .util import *


class TinychainError(Exception):
    __uri__ = URI("/error")

    def __init__(self, message):
        self.message = message

    def __json__(self):
        return {str(uri(self)): [str(self.message)]}


class BadRequest(TinychainError):
    __uri__ = uri(TinychainError) + "/bad_request"


class Forbidden(TinychainError):
    __uri__ = uri(TinychainError) + "/forbidden"


class MethodNotAllowed(TinychainError):
    __uri__ = uri(TinychainError) + "/method_not_allowed"


class NotFound(TinychainError):
    __uri__ = uri(TinychainError) + "/not_found"


class NotImplemented(TinychainError):
    __uri__ = uri(TinychainError) + "/not_implemented"


class Unauthorized(TinychainError):
    __uri__ = uri(TinychainError) + "/unauthorized"


class UnknownError(TinychainError):
    __uri__ = uri(TinychainError) + "/unknown"

