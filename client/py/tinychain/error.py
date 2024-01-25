"""Generic error types."""

from .json import to_json
from .scalar.value import String
from .uri import URI


class TinyChainError(Exception):
    """An error encountered while executing a transaction."""

    __uri__ = URI("/error")

    def __init__(self, message, params=None, **kwargs):
        if kwargs and params is not None:
            raise ValueError(f"{self.__class__.__name__} takes a Map or kwargs, not both")

        params = kwargs if kwargs else params
        self.message = String(message).render(params) if params else message

    def __json__(self):
        return {str(URI(self)): [to_json(self.message)]}

    def __ns__(self, cxt, name_hint):
        from .scalar.ref import is_op_ref

        cxt.deanonymize(self.message, name_hint + "_message")

        if is_op_ref(self.message):
            cxt.assign(self.message, name_hint + "_message")


class BadRequest(TinyChainError):
    """Error indicating receipt of a request which is badly-constructed or nonsensical."""

    CODE = 400

    __uri__ = URI(TinyChainError) + "/bad_request"


class Conflict(TinyChainError):
    """Error indicating that the requested resource is unavailable due to a lock in a different transaction."""

    CODE = 409

    __uri__ = URI(TinyChainError) + "/conflict"


class Forbidden(TinyChainError):
    """Error indicating that the requestor is not authorized to access a requested resource."""

    CODE = 403

    __uri__ = URI(TinyChainError) + "/forbidden"


class MethodNotAllowed(TinyChainError):
    """
    Error indicating that the requested resource exists,
    but does not support the request method.
    """

    CODE = 405

    __uri__ = URI(TinyChainError) + "/method_not_allowed"


class NotFound(TinyChainError):
    """Error indicating that there is no resource with the requested path and key."""

    CODE = 404

    __uri__ = URI(TinyChainError) + "/not_found"


class NotImplemented(TinyChainError):
    """
    Error indicating that the requested functionality does not exist, but is planned for implementation in the future.
    """

    CODE = 501

    __uri__ = URI(TinyChainError) + "/not_implemented"


class Timeout(TinyChainError):
    """Error indicating that the request timed out."""

    CODE = 408

    __uri__ = URI(TinyChainError) + "/timeout"


class Unauthorized(TinyChainError):
    """Error indicating that the requestor's credentials are missing or invalid."""

    CODE = 401

    __uri__ = URI(TinyChainError) + "/unauthorized"


class UnknownError(TinyChainError):
    """An internal error with no handling or recovery logic defined."""

    CODE = 500

    __uri__ = URI(TinyChainError) + "/unknown"
