from .state import Method, MethodStub


def get_method(method):
    return MethodStub(Method.Get, method)

