"""Generic error types."""

from tinychain.util import uri, URI


class TinychainError(Exception):
    """An error encountered while executing a transaction."""

    __uri__ = URI("/error")

    def __init__(self, message):
        self.message = message

    def __json__(self):
        return {str(uri(self)): [str(self.message)]}


class BadRequest(TinychainError):
    """Error indicating receipt of a request which is badly-constructed or nonsensical."""

    __uri__ = uri(TinychainError) + "/bad_request"


class Forbidden(TinychainError):
    """Error indicating that the requestor is not authorized to access a requested resource."""

    __uri__ = uri(TinychainError) + "/forbidden"


class MethodNotAllowed(TinychainError):
    """
    Error indicating that the requested resource exists,
    but does not support the request method.
    """

    __uri__ = uri(TinychainError) + "/method_not_allowed"


class NotFound(TinychainError):
    """Error indicating that there is no resource with the requested path and key."""

    __uri__ = uri(TinychainError) + "/not_found"


class NotImplemented(TinychainError):
    """
    Error indicating that the requested functionality does not exist,
    but is planned for implementation in the future.
    """

    __uri__ = uri(TinychainError) + "/not_implemented"


class Unauthorized(TinychainError):
    """Error indicating that the requestor's credentials are missing or invalid."""

    __uri__ = uri(TinychainError) + "/unauthorized"


class UnknownError(TinychainError):
    """An internal error with no handling or recovery logic defined."""

    __uri__ = uri(TinychainError) + "/unknown"

