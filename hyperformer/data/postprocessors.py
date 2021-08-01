def string_to_float(string, default=-1.):
    """Converts string to float, using default when conversion not possible."""
    try:
        return float(string)
    except ValueError:
        return default


def string_to_int(string, default=-1):
    """Converts string to int, using default when conversion not possible."""
    try:
        return int(string)
    except ValueError:
        return default


def get_post_processor(task):
    """Returns post processor required to apply on the predictions/targets
    before computing metrics for each task."""
    if task == "stsb":
        return string_to_float
    elif task in ["qqp", "cola", "mrpc"]:
        return string_to_int
    else:
        return None
