"""Generic error types."""

from tinychain.util import deanonymize, to_json, uri, URI
from tinychain.value import String


class TinyChainError(Exception):
    """An error encountered while executing a transaction."""

    __uri__ = URI("/error")

    def __init__(self, message, params=None, **kwargs):
        if kwargs and params is not None:
            raise ValueError(f"{self.__class__.__name__} takes a Map or kwargs, not both")

        params = kwargs if kwargs else params
        self.message = String(message).render(params) if params else message

    def __json__(self):
        return {str(uri(self)): [to_json(self.message)]}

    def __ns__(self, cxt):
        deanonymize(self.message, cxt)


class BadRequest(TinyChainError):
    """Error indicating receipt of a request which is badly-constructed or nonsensical."""

    __uri__ = uri(TinyChainError) + "/bad_request"


class Conflict(TinyChainError):
    """Error indicating that the requested resource is unavailable due to a lock in a different transaction."""

    __uri__ = uri(TinyChainError) + "/conflict"


class Forbidden(TinyChainError):
    """Error indicating that the requestor is not authorized to access a requested resource."""

    __uri__ = uri(TinyChainError) + "/forbidden"


class MethodNotAllowed(TinyChainError):
    """
    Error indicating that the requested resource exists,
    but does not support the request method.
    """

    __uri__ = uri(TinyChainError) + "/method_not_allowed"


class NotFound(TinyChainError):
    """Error indicating that there is no resource with the requested path and key."""

    __uri__ = uri(TinyChainError) + "/not_found"


class NotImplemented(TinyChainError):
    """
    Error indicating that the requested functionality does not exist,
    but is planned for implementation in the future.
    """

    __uri__ = uri(TinyChainError) + "/not_implemented"


class Timeout(TinyChainError):
    """Error indicating that the request timed out."""

    __uri__ = uri(TinyChainError) + "/timeout"


class Unauthorized(TinyChainError):
    """Error indicating that the requestor's credentials are missing or invalid."""

    __uri__ = uri(TinyChainError) + "/unauthorized"


class UnknownError(TinyChainError):
    """An internal error with no handling or recovery logic defined."""

    __uri__ = uri(TinyChainError) + "/unknown"

