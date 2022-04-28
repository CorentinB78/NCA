import numpy as np
from copy import copy
from .utilities import print_warning_large_error


def fixed_point_loop(
    f,
    x0,
    tol=1e-8,
    max_iter=100,
    verbose=False,
    callback_func=None,
    err_func=None,
    alpha=1.0,
    f_kwargs=None,
):
    """
    Solve fixed point problem by performing x <- f(x) loops.
    """
    n_iter = 0
    err = +np.inf
    x = copy(x0)

    if f_kwargs is None:
        f_kwargs = {}

    while err > tol and n_iter < max_iter:
        old_x = copy(x)
        x += alpha * (f(x, **f_kwargs) - x)
        n_iter += 1

        if err_func is None:
            err = np.sum(np.abs(x - old_x))
        else:
            err = err_func(x - old_x)

        if not np.isfinite(err):
            raise RuntimeError(f"Error function returned not finite value: {err}")

        if verbose:
            print(n_iter, err)

        if callback_func is not None:
            callback_func(x, n_iter)

    if verbose:
        print("Done.")
        print()

    print_warning_large_error(
        f"Fixed point loop: Poor convergence. Error={err}", err, tolw=tol, tole=1e-3
    )

    return x, err, n_iter
