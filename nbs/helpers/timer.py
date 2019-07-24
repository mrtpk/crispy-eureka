# Code taken from online video
#
# Title: 'James Powell - So you want to be a Python expert?'
# Link: https://www.youtube.com/watch?v=cKPlPJyQrt4
from time import time


def timer(func):

    def f(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print("Elapsed time to exectute {.__name__}: {}".format(func, after-before))
        return rv

    return f


def ntimes(n):
    def inner(f):
        def wrapper(*args, **kwargs):
            rv = None
            for _ in range(n):
                print('Running {.__name__}'.format(f))
                rv = f(*args, **kwargs)
            return rv
        return wrapper
    return inner
