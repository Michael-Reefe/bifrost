import os
import glob
import time
import re
import datetime
from numba import jit, njit, prange
import numpy as np


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


def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return int(stepper * number) / stepper


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


@njit
def calc_rest_frame(wave, redshift):
    """
    Calculate the rest frame wavelengths of the object from the redshift.

    :param wave: np.ndarray
        The wavelength array
    :param redshift: float
        The redshift of the object
    :return wave: np.ndarray
        The redshift-corrected wavelength array.
    """
    wave /= (1+redshift)
    return wave


@njit
def correct_extinction(wave, flux, ebv, r_v):
    """
    Deredden a flux vector using the CCM 1989 parameterization
    Returns an array of the unreddened flux

    :param wave: np.ndarray
        The wavelength vector.
    :param flux: np.ndarray
        The flux vector to be corrected.
    :param ebv: float
        E(B-V) in magintudes.
    :param r_v: optional, float
        specifies the ratio of total selective
        extinction R(V) = A(V)/E(B-V). If not specified,
        then r_v = 3.1
    :return flux: np.ndarray
        The unreddened calibrated flux array, same number of
        elements as wave

    NOTES:
    0. This function was taken from BADASS3, created by Remington Sexton, https://github.com/remingtonsexton/BADASS3.
       All notes below come from the original BADASS documentation.

    1. (From BADASS:) This function was converted from the IDL Astrolib procedure
       last updated in April 1998. All notes from that function
       (provided below) are relevant to this function

    2. (From IDL:) The CCM curve shows good agreement with the Savage & Mathis (1979)
       ultraviolet curve shortward of 1400 A, but is probably
       preferable between 1200 and 1400 A.
    3. (From IDL:) Many sightlines with peculiar ultraviolet interstellar extinction
       can be represented with a CCM curve, if the proper value of
       R(V) is supplied.
    4. (From IDL:) Curve is extrapolated between 912 and 1000 A as suggested by
       Longo et al. (1989, ApJ, 339,474)
    5. (From IDL:) Use the 4 parameter calling sequence if you wish to save the
       original flux vector.
    6. (From IDL:) Valencic et al. (2004, ApJ, 616, 912) revise the ultraviolet CCM
       curve (3.3 -- 8.0 um-1).	But since their revised curve does
       not connect smoothly with longer and shorter wavelengths, it is
       not included here.

    7. For the optical/NIR transformation, the coefficients from
       O'Donnell (1994) are used

    # >>> ccm_unred([1000, 2000, 3000], [1, 1, 1], 2 )
    array([9.7976e+012, 1.12064e+07, 32287.1])
    """
    assert wave.size == flux.size, "Wave and flux must have the same size!"

    x = 10000.0 / wave
    npts = wave.size
    a = np.zeros(npts)
    b = np.zeros(npts)

    ###############################
    # Infrared

    good = np.where((x > 0.3) & (x < 1.1))[0]
    a[good] = 0.574 * x[good] ** (1.61)
    b[good] = -0.527 * x[good] ** (1.61)

    ###############################
    # Optical & Near IR

    good = np.where((x >= 1.1) & (x < 3.3))[0]
    y = x[good] - 1.82

    c1 = np.array([1.0, 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505])
    c2 = np.array([0.0, 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])

    order = len(c1)
    poly_a = np.zeros(len(y))
    for i in prange(order):
        poly_a += c1[::-1][i] * y ** (order-1-i)
    a[good] = poly_a

    order = len(c2)
    poly_b = np.zeros(len(y))
    for j in prange(order):
        poly_b += c2[::-1][j] * y ** (order-1-j)
    b[good] = poly_b
    # a[good] = np.polyval(c1[::-1], y)
    # b[good] = np.polyval(c2[::-1], y)

    ###############################
    # Mid-UV

    good = np.where((x >= 3.3) & (x < 8))[0]
    y = x[good]
    F_a = np.zeros(good.size)
    F_b = np.zeros(good.size)
    good1 = np.where(y > 5.9)[0]

    if good1.size > 0:
        y1 = y[good1] - 5.9
        F_a[good1] = -0.04473 * y1 ** 2 - 0.009779 * y1 ** 3
        F_b[good1] = 0.2130 * y1 ** 2 + 0.1207 * y1 ** 3

    a[good] = 1.752 - 0.316 * y - (0.104 / ((y - 4.67) ** 2 + 0.341)) + F_a
    b[good] = -3.090 + 1.825 * y + (1.206 / ((y - 4.62) ** 2 + 0.263)) + F_b

    ###############################
    # Far-UV

    good = np.where((x >= 8) & (x <= 11))[0]
    y = x[good] - 8.0
    c1 = [-1.073, -0.628, 0.137, -0.070]
    c2 = [13.670, 4.257, -0.420, 0.374]

    order = len(c1)
    poly_a = np.zeros(len(y))
    for i in prange(order):
        poly_a += c1[::-1][i] * y ** (order-1-i)
    a[good] = poly_a

    order = len(c2)
    poly_b = np.zeros(len(y))
    for j in prange(order):
        poly_b += c2[::-1][j] * y ** (order-1-j)
    b[good] = poly_b
    # a[good] = np.polyval(c1[::-1], y)
    # b[good] = np.polyval(c2[::-1], y)

    # Applying Extinction Correction

    a_v = r_v * ebv
    a_lambda = a_v * (a + b / r_v)

    flux = flux * 10.0 ** (0.4 * a_lambda)

    return flux  # ,a_lambda


@njit
def make_bins(wavs):
    """
    NOTES:
        This function has been taken and adapted from the SpectRes package, https://github.com/ACCarnall/SpectRes
        to improve performance.

    ORIGINAL DOCSTRING:

    Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins.
    """
    edges = np.zeros(wavs.shape[0]+1)
    widths = np.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0])/2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2])/2
    edges[1:-1] = (wavs[1:] + wavs[:-1])/2
    widths[:-1] = edges[1:-1] - edges[:-2]

    return edges, widths


@njit
def spectres(new_wavs, spec_wavs, spec_fluxes, spec_errs, fill=None, verbose=True):

    """
    NOTES:
        This function has been taken and adapted from the SpectRes package, https://github.com/ACCarnall/SpectRes
        to improve performance.  Please note that unlike the original package, spec_errs is now a REQUIRED argument.

    ORIGINAL DOCSTRING:

    Function for resampling spectra (and optionally associated
    uncertainties) onto a new wavelength basis.

    Parameters
    ----------

    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the
        spectrum or spectra.

    spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the
        spectrum or spectra.

    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in
        spec_wavs, last dimension must correspond to the shape of
        spec_wavs. Extra dimensions before this may be used to include
        multiple spectra.

    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties
        associated with each spectral flux value.

    fill : float (optional)
        Where new_wavs extends outside the wavelength range in spec_wavs
        this value will be used as a filler in new_fluxes and new_errs.

    verbose : bool (optional)
        Setting verbose to False will suppress the default warning about
        new_wavs extending outside spec_wavs and "fill" being used.

    Returns
    -------

    new_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same
        length as new_wavs, other dimensions are the same as
        spec_fluxes.

    new_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in new_fluxes.
        Only returned if spec_errs was specified.
    """

    # Rename the input variables for clarity within the function.
    old_wavs = spec_wavs
    old_fluxes = spec_fluxes
    old_errs = spec_errs

    # Make arrays of edge positions and widths for the old and new bins

    old_edges, old_widths = make_bins(old_wavs)
    new_edges, new_widths = make_bins(new_wavs)

    # Generate output arrays to be populated
    new_fluxes = np.zeros(np.shape(old_fluxes[..., 0]) + new_wavs.shape)

    if old_errs.shape != old_fluxes.shape:
        raise ValueError("Spec_errs must be the same shape "
                         "as spec_fluxes.")
    else:
        new_errs = np.copy(new_fluxes)

    start = 0
    stop = 0
    warned = False

    # Calculate new flux and uncertainty values, looping over new bins
    for j in range(new_wavs.shape[0]):

        # Add filler values if new_wavs extends outside of spec_wavs
        if (new_edges[j] < old_edges[0]) or (new_edges[j+1] > old_edges[-1]):
            new_fluxes[..., j] = fill

            new_errs[..., j] = fill

            if (j == 0 or j == new_wavs.shape[0]-1) and verbose and not warned:
                warned = True
                print("\nSpectres: new_wavs contains values outside the range "
                      "in spec_wavs, new_fluxes and new_errs will be filled "
                      "with the value set in the 'fill' keyword argument. \n")
            continue

        # Find first old bin which is partially covered by the new bin
        while old_edges[start+1] <= new_edges[j]:
            start += 1

        # Find last old bin which is partially covered by the new bin
        while old_edges[stop+1] < new_edges[j+1]:
            stop += 1

        # If new bin is fully inside an old bin start and stop are equal
        if stop == start:
            new_fluxes[..., j] = old_fluxes[..., start]
            if old_errs is not None:
                new_errs[..., j] = old_errs[..., start]

        # Otherwise multiply the first and last old bin widths by P_ij
        else:
            start_factor = ((old_edges[start+1] - new_edges[j])
                            / (old_edges[start+1] - old_edges[start]))

            end_factor = ((new_edges[j+1] - old_edges[stop])
                          / (old_edges[stop+1] - old_edges[stop]))

            old_widths[start] *= start_factor
            old_widths[stop] *= end_factor

            # Populate new_fluxes spectrum and uncertainty arrays
            f_widths = old_widths[start:stop+1]*old_fluxes[..., start:stop+1]
            new_fluxes[..., j] = np.sum(f_widths, axis=-1)
            new_fluxes[..., j] /= np.sum(old_widths[start:stop+1])

            e_wid = old_widths[start:stop+1]*old_errs[..., start:stop+1]

            new_errs[..., j] = np.sqrt(np.sum(e_wid**2, axis=-1))
            new_errs[..., j] /= np.sum(old_widths[start:stop+1])

            # Put back the old bin widths to their initial values
            old_widths[start] /= start_factor
            old_widths[stop] /= end_factor

    # If errors were supplied return both new_fluxes and new_errs.
    return new_fluxes, new_errs
