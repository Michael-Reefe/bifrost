import os
import glob
import time


def get_filepaths_from_parent(parentdir, ftype):
    """
    Get a list of all the paths to a certain file type given a parent directory.
    :param parentdir: str
        The parent directory to look for files in
    :param ftype: str
        The type of file to look for, e.g. "txt"
    :return: list
        A list of filepaths to each file within parentdir of type ftype.
    """
    files = glob.glob(os.path.join(parentdir, '**', '*.'+ftype), recursive=True)
    files.sort()
    return files


# Make a wrapper to time each call of a function
def timer(name=None):

    def inner_timer(func, name=None):
        if not name:
            name = f'{func.__name__!r}'

        def wrapper(*args, **kwargs):
            units = 'seconds'
            start = time.monotonic()
            x = func(*args, **kwargs)
            end = time.monotonic()

            elapsed = end-start
            if elapsed > 120:
                units = 'minutes'
                elapsed /= 60
                if elapsed > 60:
                    units = 'hours'
                    elapsed /= 60
                    if elapsed > 72:
                        units = 'days'
                        elapsed /= 24
            print(name + f' finished in {elapsed:.4f} ' + units)
            return x

        return wrapper
    return lambda f: inner_timer(f, name=name)