from .reflect import Class, Method, MethodStub


def class_def(cls):
    return Class(cls)


def get_method(method):
    return MethodStub(Method.Get, method)


def put_method(method):
    return MethodStub(Method.Put, method)


def post_method(method):
    return MethodStub(Method.Post, method)


def delete_method(method):
    return MethodStub(Method.Delete, method)

