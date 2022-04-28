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
