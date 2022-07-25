import numpy as np


def print_warning_large_error(message, error_value, tolw=1e-2, tole=1e-1):
    """
    Print a warning if `error_value` is too large.

    Print a warning if error is larger than `tolw`.
    Print an emphasized message if it is larger than `tole`.
    """
    if abs(error_value) > tole:
        print("XXX " + message)
    elif abs(error_value) > tolw:
        print("/!\ " + message)


def symmetrize(coord, values, center, function=None, axis=-1, snap=1e-10):
    coord_out = np.asarray(coord).copy()
    values_out = np.moveaxis(np.asarray(values).copy(), axis, -1)

    if len(coord_out) != values_out.shape[-1]:
        raise ValueError("This axis does not match the number of coordinates!")

    snap = np.abs(snap)
    s = slice(None, None, -1)
    if function is None:
        function = lambda x: x
    if coord_out[0] + snap < center < coord_out[-1] - snap:
        raise ValueError("center is within the coordinate range")
    elif center <= coord_out[0] + snap:
        if np.abs(center - coord_out[0]) <= snap:
            s = slice(None, 0, -1)

        coord_out = np.concatenate((-coord_out[s] + 2 * center, coord_out))
        values_out = np.concatenate((function(values_out[..., s]), values_out), axis=-1)

    elif center >= coord_out[-1] - snap:
        if np.abs(center - coord_out[-1]) <= snap:
            s = slice(-2, None, -1)

        coord_out = np.concatenate((coord_out, -coord_out[s] + 2 * center))
        values_out = np.concatenate((values_out, function(values_out[..., s])), axis=-1)

    return coord_out, np.moveaxis(values_out, -1, axis)
