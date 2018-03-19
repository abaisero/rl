from functools import wraps, update_wrapper


def fire_before(signal):
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            signal.send(func)
            return func(*args, **kwargs)
        return inner
    return outer


def fire_after(signal):
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            signal.send(func, result=result)
            return result
        return inner
    return outer
