import time
from functools import wraps

# Set to False to disable per-function timing output
timer_flag = False


def timer(func):
    """Decorator: prints wall-clock time of *func* when timer_flag is True."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if timer_flag:
            t0     = time.perf_counter()
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {time.perf_counter() - t0:.4f} s")
            return result
        return func(*args, **kwargs)
    return wrapper