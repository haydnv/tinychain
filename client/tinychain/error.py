
class TinychainError(Exception):
    PATH = "/error"


class BadRequest(TinychainError):
    PATH = TinychainError.PATH + "/bad_request"


class Forbidden(TinychainError):
    PATH = TinychainError.PATH + "/forbidden"


class MethodNotAllowed(TinychainError):
    PATH = TinychainError.PATH + "/method_not_allowed"


class NotFound(TinychainError):
    PATH = TinychainError.PATH + "/not_found"


class NotImplemented(TinychainError):
    PATH = TinychainError.PATH + "/not_implemented"


class Unauthorized(TinychainError):
    PATH = TinychainError.PATH + "/unauthorized"


class UnknownError(TinychainError):
    PATH = TinychainError.PATH + "/unknown"

