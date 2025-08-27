version = 'developer version'

import os
import time

def measure_time(path=None, mode="a"):          # outer layer
    """
    Decorator factory.
    If *path* is a string, each call is appended there.
    If *path* is None, the message is printed.
    Writtent by Chat GPT with following prompts:
        "I want to measure the execution time of functions, by using one unique function and decorators but I don't remember exactly how"
        "I want write the execution time of a function somewhere. It is possible to add a path argument to "@measure_time" calls ? "
    """

    def decorator(func):                        # real decorator
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - t0

            stamp = time.strftime("%Y‑%m‑%d %H:%M:%S")
            line  = f"{stamp}  {func.__name__}: {elapsed:.6f}s\n"

            if path:                            # write to the chosen place
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, mode) as fh:
                    fh.write(line)
            else:                               # fallback: print to stdout
                print(line.rstrip())

            return result

        # keep some metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__  = func.__doc__
        return wrapper

    # ----- Support bare @measure_time ----------
    # If the first call came with a function instead of a path, we were used as
    # "@measure_time" rather than "@measure_time(...)".  Detect & swap.
    if callable(path):
        func  = path          # the "path" argument is actually the function
        path  = None          # behave as if no path was supplied
        return decorator(func)

    return decorator