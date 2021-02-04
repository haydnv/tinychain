from .reflect import Class, Method, MethodStub


def class_def(cls):
    return Class(cls)


def get_method(method):
    return MethodStub(Method.Get, method)

