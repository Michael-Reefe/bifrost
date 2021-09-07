# Internal python modules
import os
import time
import pickle
import json
import copy
import gc

# External packages
import tqdm
import numpy as np
from numba import jit, njit, prange
import matplotlib.pyplot as plt
import plotly.subplots
import plotly.graph_objects

import astropy.coordinates
import astropy.time
import astropy.io.fits
import astropy.units as u
import astropy.constants as c
import astropy.convolution

import astroquery.irsa_dust
import spectres

# Bifrost packages
import bifrost.utils as utils
import bifrost.filters as bfilters


class Spectrum:

    __slots__ = ['wave', 'flux', 'error', 'redshift', 'ebv', 'name', 'output_path', '_corrected']

    def __init__(self, wave, flux, error, redshift=None, ebv=None, name='Generic', output_path=None):
        """
        A class containing data and attributes for a single spectrum of an astronomical object.

        :param wave: np.ndarray
            Array of the observed wavelength range of the spectrum, not corrected for redshift, in angstroms.
        :param flux: np.ndarray
            Array of the observed flux of the object over the wavelength range, not corrected for extinction, in
            10^-17 erg cm^-2 s^-1 angstrom^-1.
        :param error: np.ndarray
            Array of the error associated with the observed flux, also in units of 10^-17 erg cm^-2 s^-1 angstrom^-1.
        :param redshift: float
            Redshift of the object in units of c.
            ***IMPORTANT***
            If none is provided, it is assumed the given wavelength is already corrected to be in the rest frame of the
            source.
        :param ebv: float
            Extinction (B-V) color of the object in mag.
            ***IMPORTANT***
            If none is provided, it is assumed the given flux is already corrected for galactic extinction.
        :param name: str
            An identifier for the object spectrum.
        :param output_path: str
            A directory where saved files will default to if none is given.
        """
        # Observed wavelength array in angstroms
        self.wave = wave
        # Observed flux array in 10^-17 * CGS units
        self.flux = flux
        # Flux error in 10^-17 * CGS units
        self.error = error
        # Redshift in units of c
        self.redshift = redshift
        # Extinction in (B-V) magnitudes
        self.ebv = ebv
        # Name
        self.name = name
        # Output path
        if not output_path:
            output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', self.name)
        self.output_path = output_path
        # Verify that everything makes sense
        self._verify()
        self._corrected = False

    def __repr__(self):
        s = '########################################################################################\n'
        s += f"{self.name} Spectrum \n"
        s += '########################################################################################\n'
        corrstr = "(corrected)" if self._corrected else "(uncorrected)"
        s += f"Wavelength range " + corrstr + f": \t {np.min(self.wave)} - {np.max(self.wave)} angstroms\n"
        s += f"Flux range:                        \t {np.max(np.concatenate(([np.min(self.flux)], [0.0])))} - {np.max(self.flux)} * 10^-17 erg cm^-2 s^-1 angstrom^-1\n"
        s += f"Redshift:                          \t z = {self.redshift}\n"
        s += f"Extinction:                        \t E(B-V) = {self.ebv}\n"
        return s

    def _verify(self):
        """
        Verify that the information in the object makes sense.  i.e. the size of wave, flux, and error are all the same.

        :return None:
        """
        assert self.wave.size == self.flux.size == self.error.size, \
            "Wave, flux, and error arrays must be the same size!"

    def apply_corrections(self, r_v=3.1):
        """
        Apply all corrections to the spectrum: redshift and extinction

        :param r_v: float
            extinction ratio A(V)/E(B-V), default 3.1
        :return None:
        """
        if not self._corrected:
            if self.redshift:
                self.wave = self.calc_rest_frame(self.wave, self.redshift)
            if self.ebv:
                self.flux = self.correct_extinction(self.wave, self.flux, self.ebv, r_v)
            self._corrected = True

    @property
    def corrected(self):
        return self._corrected

    @corrected.setter
    def corrected(self, value):
        raise ValueError("The 'corrected' property may not be manually set!")

    @corrected.deleter
    def corrected(self):
        raise ValueError("The 'corrected' property may not be deleted!")

    @staticmethod
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

    @staticmethod
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

        # order = len(c1)
        # poly_a = np.zeros(len(y))
        # for i in prange(order):
        #     poly_a += c1[::-1][i] * y ** (order-1-i)
        # a[good] = poly_a
        #
        # order = len(c2)
        # poly_b = np.zeros(len(y))
        # for j in prange(order):
        #     poly_b += c2[::-1][j] * y ** (order-1-j)
        # b[good] = poly_b
        a[good] = np.polyval(c1[::-1], y)
        b[good] = np.polyval(c2[::-1], y)

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

        # order = len(c1)
        # poly_a = np.zeros(len(y))
        # for i in prange(order):
        #     poly_a += c1[::-1][i] * y ** (order-1-i)
        # a[good] = poly_a
        #
        # order = len(c2)
        # poly_b = np.zeros(len(y))
        # for j in prange(order):
        #     poly_b += c2[::-1][j] * y ** (order-1-j)
        # b[good] = poly_b
        a[good] = np.polyval(c1[::-1], y)
        b[good] = np.polyval(c2[::-1], y)

        # Applying Extinction Correction

        a_v = r_v * ebv
        a_lambda = a_v * (a + b / r_v)

        flux = flux * 10.0 ** (0.4 * a_lambda)

        return flux  # ,a_lambda

    def plot(self, convolve_width=3, emline_color="rebeccapurple", absorp_color="darkgoldenrod", overwrite=False,
             fname=None, backend='plotly', normalized=False):
        """
        Plot the spectrum.

        :param convolve_width: optional, int
            The width of convolution performed before plotting the spectrum with a Box1DKernel
        :param emline_color: optional, str
            If backend is pyplot, this specifies the color of emission lines plotted.  Default is 'rebeccapurple'.
        :param absorp_color: optional, str
            If backend is pyplot, this specifies the color of absorption lines plotted.  Default is 'darkgoldenrod'.
        :param overwrite: optional, bool
            If true, overwrites the file if it already exists.  Otherwise it is not replotted.  Default is false.
        :param fname: optional, str
            The path and file name to save the plot to.
        :param backend: optional, str
            May be 'pyplot' to use the pyplot module or 'plotly' to use the plotly module for plotting.  Default is 'plotly'.
        :param normalized: optional, bool
            If true, the y axis units are displayed as "normalized".  Otherwise, they are displayed as "10^-17 erg cm^-2 s^-1 angstrom^-1".
            Default is false.
        :return None:
        """
        # Make sure corrections have been applied
        self.apply_corrections()
        if not fname:
            fname = os.path.join(self.output_path, self.name.replace(' ', '_')+'.spectrum') + ('.pdf', '.html')[backend == 'plotly']
        if os.path.exists(fname) and not overwrite:
            return

        # Convolve the spectrum
        kernel = astropy.convolution.Box1DKernel(convolve_width)
        spectrum = astropy.convolution.convolve(self.flux, kernel)
        error = astropy.convolution.convolve(self.error, kernel)
        spectrum[spectrum<0.] = 0.
        error[error<0.] = 0.

        if backend == 'pyplot':
            # Plot the spectrum and error
            fig, ax = plt.subplots(figsize=(20, 10))
            linewidth = .5
            linestyle = '--'
            ax.plot(self.wave, spectrum, '-', color='k', lw=linewidth)
            ax.fill_between(self.wave, spectrum-error, spectrum+error, color='mediumaquamarine', alpha=0.5)

            # Plot emission and absorption lines

            # OVI, Ly-alpha, NV, OI, CII, SiIV, SiIV/OIV, CIV, HeII
            # OIII, AlIII, CIII, CII, NeIV, MgII, NeV, NeVI, [OII]
            # [OII], H-delta, H-gamma, [OIII], H-beta, [OIII], [OIII], [OI], [OI]
            # [FeX], [NII], H-alpha, [NII], [SII], [SII], [FeXI]
            emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                                1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                                3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                                6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
            for line in emlines:
                ax.axvline(line, color=emline_color, lw=linewidth, linestyle=linestyle, alpha=0.5)

            # Ca K, Ca H, Mg1b, Na, CaII, CaII, CaII
            abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
            for line in abslines:
                ax.axvline(line, color=absorp_color, lw=linewidth, linestyle=linestyle, alpha=0.5)

            # Set up axis labels and formatting
            fontsize = 20
            ax.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)', fontsize=fontsize)
            if not normalized:
                ax.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)', fontsize=fontsize)
            else:
                ax.set_ylabel(r'$f_\lambda$ (normalized)', fontsize=fontsize)
            ax.set_title('%s, $z=%.3f$' % (self.name, self.redshift), fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize-2)
            ax.set_xlim(np.nanmin(self.wave), np.nanmax(self.wave))
            ax.set_ylim(0., np.nanmax(spectrum))

            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        elif backend == 'plotly':
            fig = plotly.subplots.make_subplots(rows=1, cols=1)
            linewidth = .5
            fig.add_trace(plotly.graph_objects.Scatter(x=self.wave, y=spectrum, line=dict(color='black', width=linewidth),
                                                       name='Data', showlegend=False))
            fig.add_trace(plotly.graph_objects.Scatter(x=self.wave, y=spectrum+error,
                                                       line=dict(color='#60dbbd', width=0), fillcolor='rgba(96, 219, 189, 0.6)',
                                                       name='Upper Bound', showlegend=False))
            fig.add_trace(plotly.graph_objects.Scatter(x=self.wave, y=spectrum-error,
                                                       line=dict(color='#60dbbd', width=0), fillcolor='rgba(96, 219, 189, 0.6)',
                                                       fill='tonexty', name='Lower Bound', showlegend=False))
            emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                                1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                                3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                                6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
            abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
            for line in emlines:
                fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#663399')
            for line in abslines:
                fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#d1c779')
            if not normalized:
                y_title = 'f<sub>&#955;</sub> (10<sup>-17</sup> erg cm<sup>-2</sup> s<sup>-1</sup> &#8491;<sup>-1</sup>)'
            else:
                y_title = 'f<sub>&#955;</sub> (normalized)'
            fig.update_layout(
                yaxis_title=y_title,
                xaxis_title='&#955;<sub>rest</sub> (&#8491;)',
                title='%s, z=%.3f' % (self.name, self.redshift),
                hovermode='x'
            )
            fig.update_xaxes(
                range=(np.nanmin(self.wave), np.nanmax(self.wave)),
                constrain='domain'
            )
            fig.update_yaxes(
                range=(0, np.nanmax(spectrum)+.3),
                constrain='domain'
            )
            fig.write_html(fname)


    def save_pickle(self):
        """
        Save the object contents to a pickle file

        :return None:
        """
        with open(os.path.join(self.output_path, self.name.replace(' ', '_')+'.data.pkl'), 'wb') as handle:
            pickle.dump(self, handle)

    @classmethod
    def from_fits(cls, filepath, name):
        """
        Create a spectrum object from a fits file
        This function was adapted from BADASS3, created by Remington Sexton, https://github.com/remingtonsexton/BADASS3.

        :param filepath: str
            The path to the fits file
        :param name: str
            The name of the spectrum.
        :return cls: Spectrum
            The Spectrum object created from the fits file.
        """
        # Load the data
        with astropy.io.fits.open(filepath) as hdu:

            specobj = hdu[2].data
            z = specobj['z'][0]
            try:
                ra = hdu[0].header['RA']
                dec = hdu[0].header['DEC']
            except:
                ra = specobj['PLUG_RA'][0]
                dec = specobj['PLUG_DEC'][0]

            t = hdu[1].data

            # Unpack the spectra
            flux = t['flux']
            wave = np.power(10, t['loglam'])
            error = np.sqrt(1 / t['ivar'])
            # and_mask = t['and_mask']

        hdu.close()
        del hdu
        del t
        del specobj
        gc.collect()

        ### Interpolating over bad pixels ###
        bad = np.where(~np.isfinite(flux) & ~np.isfinite(error))[0]
        # error[bad] = np.nanmedian(error)

        # Insert additional nans next to bad pixels
        def insert_nans(spec, bad):
            all_bad = np.unique(np.concatenate((bad-1, bad, bad+1)))
            all_bad = np.array([ab for ab in all_bad if ab > 0 and ab < len(spec)])
            try:
                spec[all_bad] = np.nan
                return spec
            except:
                return spec

        def nan_helper(spec):
            return np.isnan(spec), lambda z: z.nonzero()[0]

        flux = insert_nans(flux, bad)
        nans, x = nan_helper(flux)
        flux[nans] = np.interp(x(nans), x(~nans), flux[~nans])

        error = insert_nans(error, bad)
        nans, x = nan_helper(error)
        error[nans] = np.interp(x(nans), x(~nans), error[~nans])

        coord = astropy.coordinates.SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='fk5')
        try:
            table = astroquery.irsa_dust.IrsaDust.get_query_table(coord, section='ebv')
            ebv = table['ext SandF mean'][0]
        except:
            ebv = 0.04

        # Convert to little-endian so numba works properly
        if wave.dtype == '>f4':
            wave.byteswap(inplace=True)
            wave.dtype = '<f4'
        if flux.dtype == '>f4':
            flux.byteswap(inplace=True)
            flux.dtype = '<f4'
        if error.dtype == '>f4':
            error.byteswap(inplace=True)
            error.dtype = '<f4'

        return cls(wave, flux, error, redshift=z, ebv=ebv, name=name)


class Spectra(dict):

    """
    An extension of the base Python dictionary for storing Spectrum objects.
    """

    default_keys = Spectrum.__slots__

    def to_numpy(self, keys=None):
        """
        Unpacks values to a dict of numpy arrays.

        :param keys: str, iterable
            A list of strings containing the keys to unpack, defaults to None for all keys.

        :return out: dict
            A dictionary containing the numpy arrays.
        """
        if keys is None:
            keys = self.default_keys
        else:
            t = type(keys)
            if t is str:
                keys = [keys]
        out = {}
        for key in keys:
            t = type(getattr(self[0], key))
            out[key] = np.array([getattr(self[sname], key) for sname in self], dtype=t)
        return out

    def add_spec(self, spec):
        """
        Add a spectrum to the spectra dictionary with spec.name as a key.

        :param spec: Spectrum object
            The object to add to the dictionary.
        :return: None
        """
        self[spec.name] = spec

    def correct_spectra(self, r_v=3.1):
        """
        Apply velocity and extinction corrections to all spectra in the dictionary.

        :param r_v: float, iterable
            The extinction ratio (or ratios) A(V)/E(B-V) to use in the corrections for each spectrum.
            If float, the same correction is applied to all spectra.  Default is 3.1
        :return None:
        """
        if type(r_v) not in (list, tuple, np.ndarray):
            for item in self:
                self[item].apply_corrections(r_v=r_v)
        else:
            for item, r_vi in zip(self, r_v):
                self[item].apply_corrections(r_v=r_vi)

    def plot_spectra(self, fname_root, spectra='all', backend='plotly'):
        """
        Plot a series of spectra from the dictionary.

        :param fname_root: str
            The parent directory to save all plot figures to.
        :param spectra: str, iterable
            Dictionary keys of which spectra to plot. If 'all', all are plotted.  Defaults to all.
        :return None:
        """
        print('Plotting spectra...')
        format = '.html' if backend == 'plotly' else '.pdf'
        if not os.path.exists(fname_root):
            os.makedirs(fname_root)
        if spectra == 'all' or spectra == ['all']:
            for item in tqdm.tqdm(self):
                self[item].plot(fname=os.path.join(fname_root, self[item].name.replace(' ', '_')+'.spectrum'+format),
                                backend=backend)
        else:
            slist = [self[s] for s in spectra]
            for item in tqdm.tqdm(slist):
                item.plot(fname=os.path.join(fname_root, item.name.replace(' ', '_')+'.spectrum'+format),
                          backend=backend)

    def get_spec_index(self, name):
        """
        Get the relative position in the dictionary (as an int) of a spectra given its name.

        :param name: str
            The key of the item in the dictionary.
        :return: int
            The index of the key in the dictionary.
        """
        return list(self.keys()).index(name)

    def get_spec_name(self, index):
        """
        Get the name of a spectra given its position in the dictionary (as an int).

        :param index: int
            The position of the item in the dictionary.
        :return: str
            The key of the dictionary entry.
        """
        return list(self.keys())[index]

    def __setitem__(self, key, spec):
        if spec.name == 'Generic':
            spec.name = key
        super().__setitem__(key, spec)

    def __getitem__(self, key):
        t = type(key)
        if t is str or t is np.str or t is np.str_:
            return super().__getitem__(key)
        elif t is int:
            return self[list(self.keys())[key]]

    def __delitem__(self, key):
        t = type(key)
        if t is str or t is np.str or t is np.str_:
            return super().__delitem__(key)
        elif t is int:
            del self[list(self.keys())[key]]

    @property
    def corrected(self):
        for item in self:
            if not self[item].corrected:
                return False
        return True

    def __repr__(self):
        return f"A collection of {len(self)} spectra."

    def save_pickle(self, filepath):
        """
        Save the object contents to a pickle file.

        :param filepath: str
            The path to save the pickle file to.
        :return None:
        """
        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)

    def save_json(self, filepath):
        """
        Save the object contents to a json file.

        :param filepath: str
            The path to save the json file to.
        :return None:
        """
        with open(filepath, 'w') as handle:
            serializable = copy.deepcopy(self)
            for key in serializable.keys():
                serializable[key].wave = serializable[key].wave.tolist()
                serializable[key].flux = serializable[key].flux.tolist()
                serializable[key].error = serializable[key].error.tolist()
            serializable = serializable.__dict__
            serialized = json.dumps(serializable, indent=4)
            handle.write(serialized)


class Stack(Spectra):

    def __init__(self, universal_grid=None, stacked_flux=None, stacked_err=None, resampled=False, normalized=False,
                 filters=None, **options):
        """
        An extension of the Spectra class (and by extension, the dictionary) specifically for stacking purposes.

        """
        # Load in the blueprints dictionary from config.json
        config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.json')
        blueprints = json.load(open(config_path, 'r'))
        """
        r_v:              Extinction ratio A(V)/E(B-V) to calculate for.  Default = 3.1
        gridspace:        Spacing of the wavelength grid.  Default = 1
        tolerance:        Tolerance for throwing out spectra that are > tolerance angstroms apart from others.  Default = 500
        norm_region:      Wavelength bounds to use for the normalization region, with no prominent lines.  Default = None
        default_filters:  Default filters to apply to all runs.  Default = []
        """
        # Edit the blueprints dictionary with any user-specified options
        for option in options:
            if option not in blueprints:
                raise ValueError(f"The {option} key is not recognized as a configuration parameter!")
            blueprints[option] = options[option]
        # Update the object using the blueprints dictionary
        self.__dict__.update(**blueprints)
        # Filters
        if filters is None:
            filters = []
        for f in self.default_filters:
            filters.append(bfilters.Filter.from_str(f))
        self.filters = filters
        # Default object properties that will be filled in later
        self.universal_grid = universal_grid
        self.stacked_flux = stacked_flux
        self.stacked_err = stacked_err
        self.resampled = resampled
        self.normalized = normalized
        super().__init__()

    def calc_norm_region(self):
        """
        Calculate the optimal region to perform normalization. Finds the largest span of wavelengths between
        absportion lines that is also covered by all the spectra in the dictionary.  Fails if no such region can
        be found.  The region is set to the instance attribute self.norm_region.

        :return None:
        """
        emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                            1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                            3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                            6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
        abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
        lines = np.concatenate((emlines, abslines))
        lines.sort()
        diffs = np.diff(lines)
        for _ in range(len(diffs)):
            imax = np.nanargmax(diffs)
            nr0, nr1 = lines[imax], lines[imax+1]
            if self[0].wave[0] < nr0 < self[0].wave[-1] and self[0].wave[0] < nr1 < self[0].wave[-1]:
                self.norm_region = (nr0, nr1)
                return nr0, nr1
            else:
                diffs[imax] = np.nan
        raise ValueError("A normalization region could not be found!")

    def filter_spectra(self):
        """
        Go through each filter and decide if each spectrum fits its criteria. If not, the specttrum is masked out.

        :return None:
        """
        aliases = {'z': 'redshift'}
        for filter in self.filters:
            if filter.attribute in aliases:
                att = aliases[filter.attribute]
            else:
                att = filter.attribute
            removals = []
            for ispec in self:
                if not (filter.lower_bound < getattr(self[ispec], att) < filter.upper_bound):
                    print(f"WARNING: Removing spectrum {self.get_spec_index(ispec)+1}: {ispec} since it does not fulfill the criteria: {filter}")
                    removals.append(ispec)
            for r in removals:
                del self[r]

    # Allow the class to be called as a way to perform the stacking
    @utils.timer(name='Stack Procedure')
    def __call__(self):
        """
        The main procedure for stacking spectra.  Performs all necessary steps at once:
            1. Convert each spectra to their rest-frame wavelengths using their redshifts.
            2. Correct each spectra's flux for galactic extinction.
            3. Find the optimal, universal wavelength grid that is in a region shared by all spectra and with uniform spacing.
            4. Resample each spectrum in the dictionary onto the universal grid, while conserving flux and flux error.
            5. Normalize each spectrum to the median value in the normalization region.
            6. Coadd each spectrum together using 1/error^2 as the weights at each pixel value.
            7. Coadd the errors for each spectrum together again using 1/error^2 as the weights at each pixel value.

        :return self.universal_grid: np.ndarray
            The universal, uniform wavelength grid
        :return self.stacked_flux: np.ndarray
            The co-added flux.
        :return self.stacked_err: np.ndarray
            The co-added error.
        """
        self.correct_spectra()
        self.filter_spectra()
        if type(self.universal_grid) is not np.ndarray:
            self.uniform_wave_grid()
        if not self.resampled:
            self.resample()
        if not self.normalized:
            self.normalize()
        self.coadd()
        return self.universal_grid, self.stacked_flux, self.stacked_err

    def correct_spectra(self):
        """
        Spectra.correct_spectra method now using the instance attribute self.r_v as the argument

        :return None:
        """
        print('Correcting spectra to rest-frame wavelengths and adjusting for galactic extinction...')
        super().correct_spectra(r_v=self.r_v)

    def uniform_wave_grid(self):
        """
        Create a uniform grid of wavelengths with spacing gridspace, covering only the regions where all spectra in
        the dictionary overlap.

        :return None:
        """
        print('Calculating a universal wave grid...')
        wave = self.to_numpy('wave')['wave']
        wmin = -1
        wmax = 1e10
        all_names = np.array([s for s in self])
        for i, wi in enumerate(wave):
            remove = False
            imin = np.nanmin(wi)
            if imin > wmin:
                if np.abs(imin - wmin) > self.tolerance and i != 0:
                    remove = True
                else:
                    wmin = imin
            imax = np.nanmax(wi)
            if imax < wmax:
                if np.abs(imin - wmin) > self.tolerance and i != 0:
                    remove = True
                else:
                    wmax = imax
            if remove:
                print(f"WARNING: Removing spectrum {i+1}: {all_names[i]} due to insufficient wavelength coverage.")
                del self[all_names[i]]

        wave_grid = np.arange(int(wmin), int(wmax)+self.gridspace, self.gridspace)
        self.universal_grid = wave_grid

    def resample(self):
        """
        Resample the current spectra to a new, uniform wavelength grid while preserving flux and error across the
        interpolation.

        :return None:
        """
        print('Resampling spectra over a uniform wave grid...')
        for ispec in tqdm.tqdm(self):
            self[ispec].flux, self[ispec].error = spectres.spectres(self.universal_grid, self[ispec].wave, self[ispec].flux, spec_errs=self[ispec].error, fill=np.nan)
            self[ispec].wave = self.universal_grid
        self.resampled = True

    def normalize(self):
        """
        Normalize all spectra by the median of the normalization region.

        :return None:
        """
        print('Normalizing spectra...')
        if not self.norm_region:
            self.calc_norm_region()
        # Use the first spectra's wave since by this point they should all be equal anyways, to calculate the region to fit
        reg = np.where((self.norm_region[0] < self.universal_grid) & (self.universal_grid < self.norm_region[1]))[0]
        for ispec in tqdm.tqdm(self):
            med = np.nanmedian(self[ispec].flux[reg])
            self[ispec].flux /= med
            self[ispec].error /= med
        self.normalized = True

    def coadd(self):
        """
        Coadd all spectra together into a single, stacked spectrum, using 1/sigma**2 as the weights.

        :return None:
        """
        self.stacked_flux = np.zeros_like(self.universal_grid, dtype=np.float64)
        self.stacked_err = np.zeros_like(self.universal_grid, dtype=np.float64)

        print('Coadding spectra...')
        for i in tqdm.trange(len(self.universal_grid)):
            self.stacked_flux[i] = np.nansum([self[name].flux[i]/self[name].error[i]**2 for name in self]) \
                                   / np.nansum([1/self[name].error[i]**2 for name in self])

        print('Coadding errors...')
        for j in tqdm.trange(len(self.universal_grid)):
            weights = np.array([1/self[name].error[j]**2 for name in self])
            M = len(np.where(weights > 0)[0])
            variance = np.nansum([(self[name].flux[j] - self.stacked_flux[j])**2 / self[name].error[j]**2 for name in self]) \
                       / ((M-1)/M * np.nansum(weights))
            self.stacked_err[j] = np.sqrt(variance)

    def plot_stacked(self, fname, emline_color="rebeccapurple", absorp_color="darkgoldenrod", backend='plotly'):
        """
        Plot the stacked spectrum.

        :param fname: str
            The path and file name to save the figure to.
        :param emline_color: str
            If backend is 'pyplot', this specifies the color of the plotted emission lines.  Default is 'rebeccapurple'.
        :param absorp_color: str
            If backend is 'pyplot', this specifies the color of the plotted absorption lines. Default is 'darkgoldenrod'.
        :param backend: str
            May be 'pyplot' to use the pyplot module or 'plotly' to use the plotly module for plotting.  Default is
            'plotly'.
        :return None:
        """
        assert self.universal_grid is not None, "Universal grid has not yet been calculated!"
        assert self.stacked_flux is not None, "Stacked flux has not yet been calculated!"
        assert self.stacked_err is not None, "Stacked error has not yet been calculated!"

        # Plot the spectrum and error
        if backend == 'pyplot':
            fig, ax = plt.subplots(figsize=(20, 10))
            linewidth = .5
            linestyle = '--'
            ax.plot(self.universal_grid, self.stacked_flux, '-', color='k', lw=linewidth)
            ax.fill_between(self.universal_grid, self.stacked_flux - self.stacked_err, self.stacked_flux + self.stacked_err,
                            color='mediumaquamarine', alpha=0.5)

            # Plot emission and absorption lines

            # OVI, Ly-alpha, NV, OI, CII, SiIV, SiIV/OIV, CIV, HeII
            # OIII, AlIII, CIII, CII, NeIV, MgII, NeV, NeVI, [OII]
            # [OII], H-delta, H-gamma, [OIII], H-beta, [OIII], [OIII], [OI], [OI]
            # [FeX], [NII], H-alpha, [NII], [SII], [SII], [FeXI]
            emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                                1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                                3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                                6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
            for line in emlines:
                ax.axvline(line, color=emline_color, lw=linewidth, linestyle=linestyle, alpha=0.5)

            # Ca K, Ca H, Mg1b, Na, CaII, CaII, CaII
            abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
            for line in abslines:
                ax.axvline(line, color=absorp_color, lw=linewidth, linestyle=linestyle, alpha=0.5)

            # Set up axis labels and formatting
            fontsize = 20
            ax.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)', fontsize=fontsize)
            ax.set_ylabel(r'$f_\lambda$ (normalized)', fontsize=fontsize)
            ax.set_title('%s' % 'Stacked Spectrum', fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize - 2)
            ax.set_xlim(np.nanmin(self.universal_grid), np.nanmax(self.universal_grid))
            ax.set_ylim(0., np.nanmax(self.stacked_flux)+.1)

            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        elif backend == 'plotly':
            fig = plotly.subplots.make_subplots(rows=1, cols=1)
            linewidth = .5
            fig.add_trace(plotly.graph_objects.Scatter(x=self.universal_grid, y=self.stacked_flux, line=dict(color='black', width=linewidth),
                                                       name='Data', showlegend=False))
            fig.add_trace(plotly.graph_objects.Scatter(x=self.universal_grid, y=self.stacked_flux+self.stacked_err,
                                                       line=dict(color='#60dbbd', width=0), fillcolor='rgba(96, 219, 189, 0.6)',
                                                       name='Upper Bound', showlegend=False))
            fig.add_trace(plotly.graph_objects.Scatter(x=self.universal_grid, y=self.stacked_flux-self.stacked_err,
                                                       line=dict(color='#60dbbd', width=0), fillcolor='rgba(96, 219, 189, 0.6)',
                                                       fill='tonexty', name='Lower Bound', showlegend=False))
            emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                                1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                                3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                                6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
            abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
            for line in emlines:
                fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#663399')
            for line in abslines:
                fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#d1c779')
            title = 'Stacked Spectra'
            fig.update_layout(
                yaxis_title='f<sub>&#955;</sub> (normalized)',
                xaxis_title='&#955;<sub>rest</sub> (&#8491;)',
                title=title,
                hovermode='x'
            )
            fig.update_xaxes(
                range=(np.nanmin(self.universal_grid), np.nanmax(self.universal_grid)),
                constrain='domain'
            )
            fig.update_yaxes(
                range=(0, np.nanmax(self.stacked_flux)+.3),
                constrain='domain'
            )
            fig.write_html(fname)

    def plot_spectra(self, fname_root, spectra='all', backend='plotly'):
        """
        Spectra.plot_spectra but incorporates the information from self.normalized.

        """
        print('Plotting spectra...')
        format = '.html' if backend == 'plotly' else '.pdf'
        if not os.path.exists(fname_root):
            os.makedirs(fname_root)
        if spectra == 'all' or spectra == ['all']:
            for item in tqdm.tqdm(self):
                self[item].plot(fname=os.path.join(fname_root, self[item].name.replace(' ', '_')+'.spectrum'+format),
                                normalized=self.normalized, backend=backend)
        else:
            slist = [self[s] for s in spectra]
            for item in tqdm.tqdm(slist):
                item.plot(fname=os.path.join(fname_root, item.name.replace(' ', '_')+'.spectrum'+format),
                          normalized=self.normalized, backend=backend)

    def save_json(self, filepath):
        """
         Spectra.save_json but also converts universal_grid, stacked_flux, and stacked_err to lists.
        """
        with open(filepath, 'w') as handle:
            serializable = copy.deepcopy(self)
            for key in serializable.keys():
                serializable[key].wave = serializable[key].wave.tolist()
                serializable[key].flux = serializable[key].flux.tolist()
                serializable[key].error = serializable[key].error.tolist()
            serializable.universal_grid = serializable.universal_grid.tolist()
            serializable.stacked_flux = serializable.stacked_flux.tolist()
            serializable.stacked_err = serializable.stacked_err.tolist()
            serializable.filters = [str(f) for f in serializable.filters]
            serializable = serializable.__dict__
            serialized = json.dumps(serializable, indent=4)
            handle.write(serialized)

    def __repr__(self):
        s = f"A collection of {len(self)} stacked spectra.\n"
        s += f"Corrected:  \t {self.corrected}\n"
        s += f"Resampled:  \t {self.resampled}\n"
        s += f"Normalized: \t {self.normalized}\n"
        s += f"Stacked:    \t {True if type(self.stacked_flux) is np.ndarray else False}\n"
        return s
