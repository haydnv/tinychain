from .reflect import Method, MethodStub, Op


def get_method(method):
    return MethodStub(Method.Get, method)


def put_method(method):
    return MethodStub(Method.Put, method)


def post_method(method):
    return MethodStub(Method.Post, method)


def delete_method(method):
    return MethodStub(Method.Delete, method)


def get_op(op):
    return Op.Get(op)


def put_op(op):
    return Op.Put(op)


def post_op(op):
    return Op.Post(op)


def delete_op(op):
    return Op.Delete(op)

