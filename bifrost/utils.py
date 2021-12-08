import os
import glob
import time
import re
import datetime

from bifrost.maths import truncate


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
    if type(ftype) is str:
        files = glob.glob(os.path.join(parentdir, '**', '*.'+ftype), recursive=True)
    elif type(ftype) is list:
        files = []
        for ft in ftype:
            files.extend(glob.glob(os.path.join(parentdir, '**', '*.'+ft), recursive=True))
    else:
        raise ValueError('invalid type for ftype: must be str or list')
    files.sort()
    return files


def gen_datestr(time=False):
    """
    Generate a string of YYYYMMDD[_HHMMSS].

    :param time: bool
        If true, the HHMMSS portion of the string is included.  Otherwise it is not.
        Default is false.
    :return dt_string: str
        The date/time string.
    """
    now = datetime.datetime.now()
    if time:
        dt_string = now.strftime("%Y%m%d_%H%M%S")
    else:
        dt_string = now.strftime("%Y%m%d")
    return dt_string


def sexagesimal(decimal: float, precision=2) -> str:
    hh = int(decimal)
    f1 = hh if hh != 0 else 1

    extra = decimal % f1
    if f1 == 1 and decimal < 0:
        extra -= 1
    mm = int(extra * 60)
    f2 = mm if mm != 0 else 1

    extra2 = (extra * 60) % f2
    if f2 == 1 and (extra * 60) < 0:
        extra2 -= 1
    ss = extra2 * 60

    hh = abs(hh)
    mm = abs(mm)
    ss = abs(ss)

    ss = truncate(ss, precision)
    fmt = '{:02d}:{:02d}:{:0%d.%df}' % (precision+3, precision)
    sign = '-' if decimal < 0 else ''
    return sign + fmt.format(hh, mm, ss)


def decimal(sexagesimal: str) -> float:
    splitter = 'd|h|m|s|:| '
    valtup = re.split(splitter, sexagesimal)
    hh, mm, ss = float(valtup[0]), float(valtup[1]), float(valtup[2])
    if hh > 0 or valtup[0] == '+00' or valtup[0] == '00':
        return hh + mm/60 + ss/3600
    elif hh < 0 or valtup[0] == '-00':
        return hh - mm/60 - ss/3600


def coord_name(ra, dec):
    sign = '+' if dec >= 0 else ''
    ra /= 15
    ra = sexagesimal(ra, 2).replace(':', '')
    dec = sexagesimal(dec, 1).replace(':', '')
    return 'RA: '+ra+', Dec: '+sign+dec


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
