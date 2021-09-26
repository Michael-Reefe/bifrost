# Internal python modules
import os
import time
import pickle
import json
import copy
import gc
import itertools

# External packages
import tqdm
import numpy as np
import pandas as pd
from numba import jit, njit, prange
import matplotlib.pyplot as plt
import plotly.subplots
import plotly.graph_objects

import scipy.optimize

import astropy.coordinates
import astropy.time
import astropy.io.fits
import astropy.units as u
import astropy.constants as c
import astropy.convolution

import astroquery.irsa_dust
# import spectres

# Bifrost packages
import bifrost.utils as utils
import bifrost.filters as bfilters


class Spectrum:

    __slots__ = ['wave', 'flux', 'error', 'redshift', 'ra', 'dec', 'ebv', 'name', 'output_path', '_corrected', 'data']

    def __init__(self, wave, flux, error, redshift=None, ra=None, dec=None, ebv=None, name='Generic', output_path=None, **data):
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
        :param ra: float
            Right ascension in hours.
        :param dec: float
            Declination in degrees.
        :param ebv: float
            Extinction (B-V) color of the object in mag.
            ***IMPORTANT***
            If none is provided, it is assumed the given flux is already corrected for galactic extinction.
        :param name: str
            An identifier for the object spectrum.
        :param data: dict
            Any other parameters that might want to be sorted by.
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
        # Coordinates
        self.ra = ra
        self.dec = dec
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

        # Other properties that might want to be sorted by
        self.data = data

    def __repr__(self):
        s = '########################################################################################\n'
        s += f"{self.name} Spectrum \n"
        s += '########################################################################################\n'
        corrstr = "(corrected)" if self._corrected else "(uncorrected)"
        s += f"Wavelength range " + corrstr + f":       \t {np.min(self.wave)} - {np.max(self.wave)} angstroms\n"
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
                self.wave = utils.calc_rest_frame(self.wave, self.redshift)
            if self.ebv:
                self.flux = utils.correct_extinction(self.wave, self.flux, self.ebv, r_v)
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

    def _calc_agn_dist(self, bpt_1, bpt_2, ref_point):
        assert bpt_1 in self.data.keys() and bpt_2 in self.data.keys()
        # Calculate x and y distance between this object and the reference point in the BPT plane
        return self.data[bpt_1] - ref_point[0], self.data[bpt_2] - ref_point[1]

    def calc_agn_frac(self, bpt_1, bpt_2, ref_point):
        """
        Calculate the AGN fraction of an object based on BPT data.

        :param bpt_1: str
            The name of the first BPT ratio, should be defined in self.data
        :param bpt_2: str
            The name of the second BPT ratio, should be defined in self.data
        :param ref_point: tuple
            The reference point to calculate the distance from
        :return agn_frac: float
            The AGN fraction of the object, defined as 1 / distance.
        """
        dx, dy = self._calc_agn_dist(bpt_1, bpt_2, ref_point)
        d = np.hypot(dx, dy)
        # AGN fraction = 1 / distance^2
        self.data["agn_frac"] = 1/d
        return self.data["agn_frac"]

    def k01_agn_class(self, bpt_x, bpt_y):
        """
        Calculate whether the spectrum is classified as an AGN by the Kewley et al. 2001 classification scheme.

        :param bpt_x: str
            The name of the x axis BPT ratio, should be defined in self.data
        :param bpt_y: str
            The name of the y axis BPT ratio, should be defined in self.data
        :return agn: bool
            True if object is an AGN, otherwise False.
        """
        k_line = 0.61/(self.data[bpt_x]-0.47)+1.19
        agn = self.data[bpt_y] > k_line or self.data[bpt_x] >= 0.47
        self.data["agn_class"] = agn
        return self.data["agn_class"]

    def plot(self, convolve_width=3, emline_color="rebeccapurple", absorp_color="darkgoldenrod", cline_color="cyan",
             overwrite=False, fname=None, backend='plotly', normalized=False):
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
            # OIII, AlIII, CIII, CII, NeIV, MgII, [OII]
            # [OII], H-delta, H-gamma, [OIII], H-beta, [OIII], [OIII], [OI], [OI]
            # [NII], H-alpha, [NII], [SII], [SII]
            emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                                1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3727.092,
                                3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                                6549.860, 6564.610, 6585.270, 6718.290, 6732.670])
            for line in emlines:
                ax.axvline(line, color=emline_color, lw=linewidth, linestyle=linestyle, alpha=0.5)
            # Ne V, Ne V*, Fe VII, Fe V, Fe V, Ne III (not coronal), Fe V, Fe VII, Fe VI, Fe VII, Fe VI, Fe VII, Fe XIV, Ca V, Fe VI
            # Ar X, Fe VII, Fe VII*, Fe X, Fe XI
            clines = np.array([3346.790, 3426.850, 3759, 3839, 3891, 3970, 4181, 4893, 5146, 5159, 5176, 5276, 5303, 5309, 5335,
                               5533, 5720, 6087, 6374.510, 7891.800])
            for line in clines:
                ax.axvline(line, color=cline_color, lw=linewidth*2, linestyle=linestyle, alpha=0.75)

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
            clines = np.array([3346.790, 3426.850, 3759, 3839, 3891, 3970, 4181, 4893, 5146, 5159, 5176, 5276, 5303, 5309, 5335,
                               5533, 5720, 6087, 6374.510, 7891.800])
            cline_names = np.array(
                ['[Ne V]', '[Ne V]*', '[Fe VII]', '[Fe V]', '[Fe V]', '[Ne III]', '[Fe V]', '[Fe VII]', '[Fe VI]',
                 '[Fe VII]', '[Fe VI]', '[Fe VII]', '[Fe XIV]', '[Ca V]', '[Fe VI]', '[Ar X]', '[Fe VII]', '[Fe VII]*',
                 '[Fe X]', '[Fe XI]'], dtype=str
            )
            abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
            for line in emlines:
                fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#663399')
            for line, name in zip(clines, cline_names):
                fig.add_vline(x=line, line_width=2 * linewidth, line_dash='dot', line_color='#226666',
                              annotation_text=name, annotation_position='top right', annotation_font_size=12)
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
    def from_fits(cls, filepath, name, save_all_data=False):
        """
        Create a spectrum object from a fits file
        This function was adapted from BADASS3, created by Remington Sexton, https://github.com/remingtonsexton/BADASS3.

        :param filepath: str
            The path to the fits file
        :param name: str, optional
            The name of the spectrum.
        :return cls: Spectrum
            The Spectrum object created from the fits file.
        """
        # Load the data
        data_dict = {}
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

            if save_all_data:
                for key in specobj.names:
                    if key not in ('z', 'PLUG_RA', 'PLUG_DEC'):
                        data_dict[key] = specobj[key]
                for key in hdu[0].header:
                    if key not in data_dict.keys() and key not in ('RA', 'DEC'):
                        data_dict[key] = hdu[0].header[key]
                for key in hdu[1].data.names:
                    if key not in data_dict.keys() and key not in ('flux', 'loglam', 'ivar'):
                        data_dict[key] = hdu[1].data[key]
                for key in hdu[3].data.names:
                    if key not in data_dict.keys():
                        data_dict[key] = hdu[3].data[key]

        hdu.close()
        del hdu
        del t
        del specobj
        gc.collect()

        # if not name:
        #     name = utils.iauname(ra, dec)

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

        return cls(wave, flux, error, redshift=z, ra=ra/15, dec=dec, ebv=ebv, name=name, **data_dict)


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
        # elif t in (list, tuple, np.ndarray):
        #     if type(key[0]) in (str, np.str, np.str_):
        #         return [super().__getitem__(ki) for ki in key]
        #     elif type(key[0]) is int:
        #         return [self[list(self.keys())[ki]] for ki in key]

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

    def __init__(self, universal_grid=None, stacked_flux=None, stacked_err=None, filters=None, **options):
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
        if not universal_grid:
            universal_grid = []
        if not stacked_flux:
            stacked_flux = []
        if not stacked_err:
            stacked_err = []
        self.universal_grid = universal_grid
        self.stacked_flux = stacked_flux
        self.stacked_err = stacked_err
        self.binned = None
        self.binned_spec = None
        self.bin_counts = None
        self.bin_edges = None
        self.bin_log = False
        self.agn_ref_pt = None
        super().__init__()

    def calc_norm_region(self, wave_grid, binned_spec=None):
        """
        Calculate the optimal region to perform normalization. Finds the largest span of wavelengths between
        absportion lines that is also covered by all the spectra in the dictionary.  Fails if no such region can
        be found.  The region is set to the instance attribute self.norm_region.

        :param binned_spec: iterable
            A list of spectra names to use.  If None, all are used.
        :return nr0, nr1: tuple
            The left and right edges of the normalization region.
        """
        emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                            1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                            3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                            6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
        abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
        lines = np.concatenate((emlines, abslines))
        lines.sort()
        spectra = binned_spec if binned_spec is not None else [s for s in self]
        diffs = np.diff(lines)
        for _ in range(len(diffs)):
            imax = np.nanargmax(diffs)
            nr0, nr1 = lines[imax], lines[imax+1]
            if wave_grid[0] < nr0 < wave_grid[-1] and wave_grid[0] < nr1 < wave_grid[-1]:
                return nr0, nr1
            else:
                diffs[imax] = np.nan
        print("WARNING: An ideal normalization region could not be found!  Using the entire range.")
        return wave_grid[0], wave_grid[-1]

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

    def bin_spectra(self, bin_quantity, bin_size=None, nbins=None, log=False, midpoints=False, round_bins=None):
        """
        Place spectra into bins for the bin_quantity.

        :param bin_quantity: str
            The quantity that the data should be binned by.  Spectra must have appropriate values within their
            self.data dictionaries.
        :param bin_size: float
            How large each bin should be.  If specified, then do not also specify nbins.
        :param nbins: int
            The number of bins to use.  If specified, then do not also specify bin_size.
        :param log: bool
            Whether or not to take the log_10 of the bin_quantity before binning.
        :param midpoints: bool
            Whether or not to return the midpoints of each bin instead of the edges.  Default is false.
        :param round_bins: float
            Whether or not to make bins regular, i.e. round to nice numbers like the nearest 0.1 or 0.5.
            Round the bin edges to the nearest multiple of this float.
        :return binned_spec: dict
            Dictionary containing the names of the spectra within each bin.  The keys are the bin indices.
        :return bin_counts: np.ndarray(shape=(nbins,))
            The number of spectra in each bin.
        :return bin_edges: np.ndarray(shape=(nbins+1,))
            The edge values of each bin.
        """
        self.bin_log = log
        # Make sure arguments make sense
        if not bin_size and not nbins:
            raise ValueError("One of {bin_size,nbins} must be specified!")
        if bin_size and nbins:
            raise ValueError("Both of {bin_size,nbins} may not be specified!")

        # Load in the data from each spectrum and make an array of the appropriate quantity and names
        data = self.to_numpy('data')['data']
        unbinned = np.array([], dtype=np.float64)
        included = np.array([], dtype=np.str)
        for ispec in self:
            # Gather the data to bin
            i = self.get_spec_index(ispec)
            if bin_quantity not in data[i].keys():
                print(f"WARNING: bin_quantity not found in {ispec} data!  Ignoring this spectrum")
                continue
            id = data[i][bin_quantity]
            if log:
                id = np.log10(id)
            if np.isnan(id):
                print(f"WARNING: bin_quantity is {id} in {ispec} data!  Ignoring this spectrum")
                continue
            included = np.append(included, ispec)
            unbinned = np.append(unbinned, id)

        # Perform the binning
        minbin = np.nanmin(unbinned)
        maxbin = np.nanmax(unbinned)
        if bin_size:
            nbins = -int(-(maxbin-minbin) // bin_size)
        elif nbins:
            bin_size = (maxbin-minbin) / nbins
        if round_bins:
            rating = 1/round_bins
            minbin = np.floor(minbin*rating)/rating
            maxbin = np.ceil(maxbin*rating)/rating
            bin_size = np.round(bin_size*rating)/rating
            if bin_size == 0:
                bin_size = round_bins
            nbins = -int(-(maxbin-minbin) // bin_size)

        binned_spec = {i: np.array([], dtype=np.str) for i in range(nbins)}
        bin_counts = np.zeros(nbins)

        bin_edges = minbin + np.arange(0, nbins+1, 1)*bin_size
        bin_midpts = minbin + (np.arange(0, nbins, 1) + 0.5)*bin_size
        for i in range(len(included)):
            indx = int((unbinned[i] - minbin) / bin_size)
            if indx == len(binned_spec):
                indx -= 1
            binned_spec[indx] = np.append(binned_spec[indx], included[i])
            bin_counts[indx] += 1

        if midpoints:
            return binned_spec, bin_counts, bin_midpts
        return binned_spec, bin_counts, bin_edges

    def histogram_3D(self, fname_base, bin_quantities, logs, nbins=None, bin_size=None, round_bins=None, labels=None,
                     backend='plotly', colormap=None):

        binx, biny, binz = bin_quantities
        logx, logy, logz = logs
        if nbins:
            nbx, nby, nbz = nbins
        else:
            nbx, nby, nbz = None, None, None
        if bin_size:
            bsx, bsy, bsz = bin_size
        else:
            bsx, bsy, bsz = None, None, None
        if round_bins:
            rbx, rby, rbz = round_bins
        else:
            rbx, rby, rbz = None, None, None
        specx, countsx, edgex = self.bin_spectra(binx, log=logx, nbins=nbx, bin_size=bsx, round_bins=rbx)
        specy, countsy, edgey = self.bin_spectra(biny, log=logy, nbins=nby, bin_size=bsy, round_bins=rby)
        # specz, countsz, edgez = self.bin_spectra(binz, log=logz, nbins=nbz, bin_size=bsz, round_bins=rbz)

        nbx = len(countsx)
        nby = len(countsy)
        # nbz = len(countsz)
        z_array = np.zeros(shape=(nby, nbx), dtype=np.float64)
        n_array = np.zeros(shape=(nby, nbx), dtype=np.int64)
        for x,y in itertools.product(np.arange(nbx), np.arange(nby)):
            # specx[x], specy[y]
            good_spec = np.array([], dtype=np.str)
            z_spec = np.array([], dtype=np.float64)
            for spec in specx[x]:
                if spec in specy[y]:
                    good_spec = np.append(good_spec, spec)
                    zi = self[spec].data[binz]
                    if logz:
                        zi = np.log10(zi)
                    z_spec = np.append(z_spec, zi)
            z_array[y, x] = np.median(z_spec)
            n_array[y, x] = good_spec.size

        if backend == 'pyplot':
            if not colormap:
                colormap = 'winter'
            fig, ax = plt.subplots(figsize=(nbx/nby*7.5+3.5, 7.5))
            mesh = ax.pcolormesh(edgex, edgey, z_array, shading='flat', cmap=colormap)

            if not labels:
                xlabel = binx if not logx else '$\\log_{10}($' + binx + '$)$'
                ylabel = biny if not logy else '$\\log_{10}($' + biny + '$)$'
                zlabel = binz if not logz else '$\\log_{10}($' + binz + '$)$'
            else:
                xlabel, ylabel, zlabel = labels
            fig.colorbar(mesh, ax=ax, label=zlabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            for x,y in itertools.product(np.arange(nbx), np.arange(nby)):
                if n_array[y, x] > 0:
                    ax.text(edgex[x]+(edgex[x+1]-edgex[x])/2, edgey[y]+(edgey[y+1]-edgey[y])/2, str(n_array[y, x]),
                            fontsize=7, horizontalalignment='center', verticalalignment='center', color='white')

            ax.set_xticks(edgex[::3])
            ax.set_yticks(edgey)
            fig.savefig(fname_base+'.pdf', dpi=300, bbox_inches='tight')
            plt.close()
        elif backend == 'plotly':
            if not colormap:
                colormap = 'plasma'
            if not labels:
                xlabel = binx if not logx else 'log<sub>10</sub>(' + binx + ')'
                ylabel = biny if not logy else 'log<sub>10</sub>(' + biny + ')'
                zlabel = binz if not logz else 'log<sub>10</sub>(' + binz + ')'
            else:
                xlabel, ylabel, zlabel = labels
            fig = plotly.graph_objects.Figure(
                data=plotly.graph_objects.Heatmap(x=edgex, y=edgey, z=z_array,
                colorbar=dict(title=zlabel), colorscale=colormap)
            )
            for x,y in itertools.product(np.arange(nbx), np.arange(nby)):
                if n_array[y, x] > 0:
                    fig.add_annotation(text=str(n_array[y, x]), xref="x", yref="y", x=edgex[x]+(edgex[x+1]-edgex[x])/2,
                                       y=edgey[y]+(edgey[y+1]-edgey[y])/2, showarrow=False, font=dict(size=7, color="white"))
            fig.update_yaxes(
                scaleratio=1,
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                ticks='inside',
                tickwidth=2,
                tickcolor='black',
                ticklen=10,
                title_text=ylabel
            )
            fig.update_xaxes(
                showgrid=False,
                zeroline=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                ticks='inside',
                tickwidth=2,
                tickcolor='black',
                ticklen=10,
                title_text=xlabel
            )
            # fig.update_layout(
                # paper_bgcolor='rgba(0,0,0,0)',
                # plot_bgcolor='rgba(0,0,0,0)'
            # )
            fig.write_html(fname_base+'.html')


    # Allow the class to be called as a way to perform the stacking
    @utils.timer(name='Stack Procedure')
    def __call__(self, bin=None, nbins=None, bin_size=None, log=False, round_bins=None, auto_norm_region=True,
                 bpt_1=None, bpt_2=None, hbin_target=3, stack_all_agns=False):
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
        self.universal_grid = []
        self.stacked_flux = []
        self.stacked_err = []
        if stack_all_agns:
            if not bpt_1 or not bpt_2:
                raise ValueError("Must specify BPT ratio keys if stacking all AGNs!")

            # Calculate the statistic to cut by and get the spectra that qualify
            spectra = np.array([], dtype=str)
            for i, ispec in enumerate(self):
                agn = self[ispec].k01_agn_class(bpt_1, bpt_2)
                if agn:
                    spectra = np.append(spectra, ispec)

            # Stack only the qualified spectra
            if len(spectra) == 0:
                self.universal_grid.append(None)
                self.stacked_flux.append(None)
                self.stacked_err.append(None)
                return

            wave_grid_b, spectra = self.uniform_wave_grid(spectra)
            self.resample(wave_grid_b, spectra)
            if auto_norm_region:
                nr0, nr1 = self.calc_norm_region(wave_grid_b, spectra)
            else:
                nr0, nr1 = self.norm_region
            self.normalize(wave_grid_b, (nr0, nr1), spectra)
            flux_b, err_b = self.coadd(wave_grid_b, spectra)
            self.universal_grid.append(wave_grid_b)
            self.stacked_flux.append(flux_b)
            self.stacked_err.append(err_b)

        elif bin:
            if bin == 'agn_frac':

                if not bpt_1 or not bpt_2:
                    raise ValueError("Must specify BPT ratio keys if binning by AGN fraction!")

                # First find the optimal point to use as a reference point
                dx = np.zeros(len(self))
                dy = np.zeros(len(self))
                for i, ispec in enumerate(self):
                    dx[i], dy[i] = self[ispec]._calc_agn_dist(bpt_1, bpt_2, ref_point=(0, 0))

                # Solve for the optimal reference point so the highest bin has at least 3 spectra
                def num_in_highest_bin(ref_point_shift, target_num, bin_size=None, nbins=None, round_bins=None):
                    ref_point = (np.nanmax(dx), np.nanmax(dy)) + ref_point_shift
                    agn_fracs = np.zeros(len(self))
                    # Now calculate the AGN fractions using this distance
                    for i, ispec in enumerate(self):
                        agn_fracs[i] = self[ispec].calc_agn_frac(bpt_1, bpt_2, ref_point=ref_point)
                    # Normalize so that the largest AGN frac is 1
                    max_frac = np.nanmax(agn_fracs)
                    agn_fracs /= max_frac

                    minbin = np.nanmin(agn_fracs)
                    maxbin = 1
                    if nbins and not bin_size:
                        bin_size = (maxbin - minbin) / nbins
                    if round_bins:
                        rating = 1 / round_bins
                        bin_size = np.round(bin_size * rating) / rating
                        if bin_size == 0:
                            bin_size = round_bins
                    counts = len(np.where(agn_fracs >= 1-bin_size)[0])

                    print('Current number in highest bin: %03d' % counts, end='\r', flush=True)
                    if counts > target_num:
                        return 0
                    else:
                        return -1 * counts

                # USE POWELL METHOD
                print('Optimizing AGN fraction reference point...')
                min_res = scipy.optimize.minimize(num_in_highest_bin, 0, args=(hbin_target, bin_size, nbins, round_bins), method='Powell')
                shift = min_res.x
                print('\n')

                # Calculate final AGN fractions from the optimized reference point
                ref_point = (np.nanmax(dx), np.nanmax(dy)) + shift
                agn_fracs = np.zeros(len(self))
                for i, ispec in enumerate(self):
                    agn_fracs[i] = self[ispec].calc_agn_frac(bpt_1, bpt_2, ref_point=ref_point)
                max_frac = np.nanmax(agn_fracs)
                for i, ispec in enumerate(self):
                    agn_fracs[i] /= max_frac
                    self[ispec].data["agn_frac"] = agn_fracs[i]
                self.agn_ref_pt = ref_point

            binned_spectra, bin_counts, bin_edges = self.bin_spectra(bin, bin_size=bin_size, nbins=nbins, log=log,
                                                                     round_bins=round_bins)
            nn = len(binned_spectra)
            for i, b in enumerate(binned_spectra):
                print(f'BIN {i+1} OF {nn}...')
                spectra = binned_spectra[b]
                if len(spectra) == 0:
                    self.universal_grid.append(None)
                    self.stacked_flux.append(None)
                    self.stacked_err.append(None)
                    continue
                wave_grid_b, spectra = self.uniform_wave_grid(spectra)
                self.resample(wave_grid_b, spectra)
                if auto_norm_region:
                    nr0, nr1 = self.calc_norm_region(wave_grid_b, spectra)
                else:
                    nr0, nr1 = self.norm_region
                self.normalize(wave_grid_b, (nr0, nr1), spectra)
                flux_b, err_b = self.coadd(wave_grid_b, spectra)
                self.universal_grid.append(wave_grid_b)
                self.stacked_flux.append(flux_b)
                self.stacked_err.append(err_b)
            self.binned = bin
            self.binned_spec = binned_spectra
            self.bin_counts = bin_counts
            self.bin_edges = bin_edges
        else:
            wave_grid, _ = self.uniform_wave_grid()
            self.resample(wave_grid)
            if auto_norm_region:
                nr0, nr1 = self.calc_norm_region(wave_grid)
            else:
                nr0, nr1 = self.norm_region
            self.normalize(wave_grid, (nr0, nr1))
            flux, err = self.coadd(wave_grid)
            self.universal_grid.append(wave_grid)
            self.stacked_flux.append(flux)
            self.stacked_err.append(err)
            self.binned = None
            self.binned_spec = None
            self.bin_counts = None
            self.bin_edges = None

    def correct_spectra(self):
        """
        Spectra.correct_spectra method now using the instance attribute self.r_v as the argument

        :return None:
        """
        print('Correcting spectra to rest-frame wavelengths and adjusting for galactic extinction...')
        super().correct_spectra(r_v=self.r_v)

    def uniform_wave_grid(self, binned_spec=None):
        """
        Create a uniform grid of wavelengths with spacing gridspace, covering only the regions where all spectra in
        the dictionary overlap.

        :param binned_spec: iterable
            A list of spectra names to use.  If None, all are used.
        :return wave_grid:
            The universal wave grid.
        """
        print('Calculating a universal wave grid...')
        all_names = binned_spec if binned_spec is not None else np.array([s for s in self], dtype=np.str)
        binned_indices = np.array([self.get_spec_index(name) for name in all_names], dtype=np.int)
        wave = self.to_numpy('wave')['wave'][binned_indices]
        wmin = -1
        wmax = 1e100
        removed_names = np.array([], dtype=np.int)
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
                removed_names = np.append(removed_names, i)

        all_names = np.delete(all_names, removed_names)
        wave_grid = np.arange(int(wmin), int(wmax)+self.gridspace, self.gridspace)
        return wave_grid, all_names

    def resample(self, wave_grid, binned_spec=None):
        """
        Resample the current spectra to a new, uniform wavelength grid while preserving flux and error across the
        interpolation.

        :param wave_grid: np.ndarray
            The grid of wavelengths to resample to.
        :param binned_spec: iterable
            A list of spectra names to use.  If None, all are used.
        :return None:
        """
        print('Resampling spectra over a uniform wave grid...')
        ss = binned_spec if binned_spec is not None else [s for s in self]
        for ispec in tqdm.tqdm(ss):
            self[ispec].flux, self[ispec].error = utils.spectres(wave_grid, self[ispec].wave, self[ispec].flux, self[ispec].error, fill=np.nan)
            self[ispec].wave = wave_grid

    def normalize(self, wave_grid, norm_region, binned_spec=None):
        """
        Normalize all spectra by the median of the normalization region.

        :param norm_region: tuple
            The left and right edges of wavelength to normalize by.
        :param binned_spec: iterable
            A list of spectra names to use.  If None, all are used.
        :return None:
        """
        print('Normalizing spectra...')
        # Use the first spectra's wave since by this point they should all be equal anyways, to calculate the region to fit
        reg = np.where((norm_region[0] < wave_grid) & (wave_grid < norm_region[1]))[0]
        ss = binned_spec if binned_spec is not None else [s for s in self]
        for ispec in tqdm.tqdm(ss):
            self[ispec].flux, self[ispec].error = self._norm(self[ispec].flux, self[ispec].error, reg)
            med = np.nanmedian(self[ispec].flux[reg])
            self[ispec].flux /= med
            self[ispec].error /= med

    @staticmethod
    @njit
    def _norm(data, error, region):
        med = np.nanmedian(data[region])
        data /= med
        error /= med
        return data, error

    def coadd(self, wave_grid, binned_spec=None):
        """
        Coadd all spectra together into a single, stacked spectrum, using 1/sigma**2 as the weights.

        :param binned_spec: iterable
            A list of spectra names to use.  If None, all are used.
        :return None:
        """
        stacked_flux = np.zeros_like(wave_grid, dtype=np.float64)
        stacked_err = np.zeros_like(wave_grid, dtype=np.float64)

        print('Coadding spectra...')
        ss = binned_spec if binned_spec is not None else [s for s in self]
        for i in tqdm.trange(len(wave_grid)):
            flux_i = np.array([self[name].flux[i] for name in ss])
            err_i = np.array([self[name].error[i] for name in ss])
            if len(ss) > 1:
                stacked_flux[i], stacked_err[i] = self._coadd_flux_err(flux_i, err_i)
            else:
                stacked_flux[i], stacked_err[i] = flux_i, err_i

        return stacked_flux, stacked_err

    @staticmethod
    @njit
    def _coadd_flux_err(flux, error):
        weights = 1 / error ** 2
        M = len(np.where(weights > 0)[0])
        if np.isnan(flux).all() or np.isnan(error).all() or M <= 1:
            return np.nan, np.nan
        stacked_flux = np.nansum(flux*weights) / np.nansum(weights)
        stacked_err = np.sqrt(
            np.nansum((flux - stacked_flux)**2*weights) / ((M-1)/M * np.nansum(weights))
        )
        # stacked_err = np.sqrt(1/np.nansum(weights))
        return stacked_flux, stacked_err

    def plot_stacked(self, fname_base, emline_color="rebeccapurple", absorp_color="darkgoldenrod", cline_color="cyan",
                     backend='plotly'):
        """
        Plot the stacked spectrum.

        :param fname_base: str
            The path and file name to save the figure to.
        :param bin_num: int
            Which stacked bin should be plotted.  Default is 0, or the first bin (if data is not binned, then it's the stack
            of all spectra).
        :param emline_color: str
            If backend is 'pyplot', this specifies the color of the plotted emission lines.  Default is 'rebeccapurple'.
        :param absorp_color: str
            If backend is 'pyplot', this specifies the color of the plotted absorption lines. Default is 'darkgoldenrod'.
        :param backend: str
            May be 'pyplot' to use the pyplot module or 'plotly' to use the plotly module for plotting.  Default is
            'plotly'.
        :return None:
        """
        format0 = '.html' if backend == 'plotly' else '.pdf'
        assert self.universal_grid is not None, "Universal grid has not yet been calculated!"
        assert self.stacked_flux is not None, "Stacked flux has not yet been calculated!"
        assert self.stacked_err is not None, "Stacked error has not yet been calculated!"

        for bin_num in range(len(self.universal_grid)):
            format = '_' + str(bin_num) + format0
            fname = fname_base + format
            wave = self.universal_grid[bin_num]
            flux = self.stacked_flux[bin_num]
            err = self.stacked_err[bin_num]
            if wave is None or flux is None or err is None:
                continue

            # Plot the spectrum and error
            if backend == 'pyplot':
                fig, ax = plt.subplots(figsize=(20, 10))
                linewidth = .5
                linestyle = '--'
                ax.plot(wave, flux, '-', color='k', lw=linewidth)
                ax.fill_between(wave, flux-err, flux+err, color='mediumaquamarine', alpha=0.5)

                # Plot emission and absorption lines

                # OVI, Ly-alpha, NV, OI, CII, SiIV, SiIV/OIV, CIV, HeII
                # OIII, AlIII, CIII, CII, NeIV, MgII, NeV, NeVI, [OII]
                # [OII], H-delta, H-gamma, [OIII], H-beta, [OIII], [OIII], [OI], [OI]
                # [FeX], [NII], H-alpha, [NII], [SII], [SII], [FeXI]
                emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                                    1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3727.092,
                                    3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                                    6549.860, 6564.610, 6585.270, 6718.290, 6732.670])

                for line in emlines:
                    ax.axvline(line, color=emline_color, lw=linewidth, linestyle=linestyle, alpha=0.5)

                # Ne V, Ne V*, Fe VII, Fe V, Fe V, Ne III (not coronal), Fe V, Fe VII, Fe VI, Fe VII, Fe VI, Fe VII, Fe XIV, Ca V, Fe VI
                # Ar X, Fe VII, Fe VII*, Fe X, Fe XI
                clines = np.array(
                    [3346.790, 3426.850, 3759, 3839, 3891, 3970, 4181, 4893, 5146, 5159, 5176, 5276, 5303, 5309, 5335,
                     5533, 5720, 6087, 6374.510, 7891.800])
                for line in clines:
                    ax.axvline(line, color=cline_color, lw=linewidth * 2, linestyle=linestyle, alpha=0.75)

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
                ax.set_xlim(np.nanmin(wave), np.nanmax(wave))
                ax.set_ylim(0., np.nanmax(flux)+.3)

                fig.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close()
            elif backend == 'plotly':
                fig = plotly.subplots.make_subplots(rows=1, cols=1)
                linewidth = .5
                fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=flux, line=dict(color='black', width=linewidth),
                                                           name='Data', showlegend=False))
                fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=flux+err,
                                                           line=dict(color='#60dbbd', width=0), fillcolor='rgba(96, 219, 189, 0.6)',
                                                           name='Upper Bound', showlegend=False))
                fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=flux-err,
                                                           line=dict(color='#60dbbd', width=0), fillcolor='rgba(96, 219, 189, 0.6)',
                                                           fill='tonexty', name='Lower Bound', showlegend=False))
                emlines = np.array([1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                                    1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3727.092,
                                    3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                                    6549.860, 6564.610, 6585.270, 6718.290, 6732.670])
                clines = np.array(
                    [3346.790, 3426.850, 3759, 3839, 3891, 3970, 4181, 4893, 5146, 5159, 5176, 5276, 5303, 5309, 5335,
                     5533, 5720, 6087, 6374.510, 7891.800])
                cline_names = np.array(
                    ['[Ne V]', '[Ne V]*', '[Fe VII]', '[Fe V]', '[Fe V]', '[Ne III]', '[Fe V]', '[Fe VII]', '[Fe VI]',
                     '[Fe VII]', '[Fe VI]', '[Fe VII]', '[Fe XIV]', '[Ca V]', '[Fe VI]', '[Ar X]', '[Fe VII]', '[Fe VII]*',
                     '[Fe X]', '[Fe XI]'], dtype=str
                )
                abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
                for line in emlines:
                    fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#663399')
                for line, name in zip(clines, cline_names):
                    fig.add_vline(x=line, line_width=2*linewidth, line_dash='dot', line_color='#226666',
                                  annotation_text=name, annotation_position='top right', annotation_font_size=12)
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
                    range=(np.nanmin(wave), np.nanmax(wave)),
                    constrain='domain'
                )
                fig.update_yaxes(
                    range=(0, np.nanmax(flux)+.3),
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
                                normalized=True, backend=backend)
        else:
            slist = [self[s] for s in spectra]
            for item in tqdm.tqdm(slist):
                item.plot(fname=os.path.join(fname_root, item.name.replace(' ', '_')+'.spectrum'+format),
                          normalized=True, backend=backend)

    def plot_hist(self, fname_base, plot_log=False, backend='plotly'):
        """
        Plot a histogram of the spectra in each bin.

        :param fname_base: str
            File name pattern.
        :param backend: str
            Whether to use matplotlib or plotly to plot stuff
        :return None:
        """
        format = '.html' if backend == 'plotly' else '.pdf'
        fname = fname_base + format
        widths = np.diff(self.bin_edges)
        midpts = (self.bin_edges[:-1] + self.bin_edges[1:])/2
        nbins = len(widths)
        if backend == 'pyplot':
            fig, ax = plt.subplots()
            ax.bar(midpts, self.bin_counts, widths, align='center', color='rebeccapurple', label='$n_{\\rm bins} = %d$' % nbins,
                   log=plot_log)
            xlabel = '$\\log_{10}($' + self.binned + '$)$' if self.bin_log else self.binned
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Number in bin')
            ax.legend()
            ax.set_xticks(self.bin_edges)
            fig.savefig(fname, dpi=300, bbox_inches='tight')
        elif backend == 'plotly':
            fig = plotly.graph_objects.Figure(data=plotly.graph_objects.Bar(x=midpts, y=self.bin_counts))
            fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                              marker_line_width=0.0, opacity=0.8)
            xlabel = 'log<sub>10</sub>(' + self.binned + ')' if self.bin_log else self.binned
            fig.update_layout(
                xaxis_title=xlabel,
                yaxis_title='Number in bin',
                hovermode='x',
                xaxis=dict(
                    tickmode='array',
                    tickvals=self.bin_edges
                )
            )
            if plot_log:
                fig.update_yaxes(
                    type="log",
                )
            fig.write_html(fname)

    def plot_agn(self, fname_base, bpt_x, bpt_y, bpt_xerr=None, bpt_yerr=None, labels=None, backend='plotly'):
        format = '.html' if backend == 'plotly' else '.pdf'
        fname = fname_base + format
        data = np.array([(self[i].data[bpt_x], self[i].data[bpt_y], self[i].data['agn_frac']) for i in range(len(self))])
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        xerr = yerr = None
        if bpt_xerr:
            xerr = np.array([self[i].data[bpt_xerr] for i in range(len(self))])
        if bpt_yerr:
            yerr = np.array([self[i].data[bpt_yerr] for i in range(len(self))])
        if labels:
            xl, yl = labels
        else:
            xl, yl = bpt_x, bpt_y
        k01_x = np.linspace(np.nanmin(x), np.min([np.nanmax(x), 0.469]), 100)
        k01_y = 0.61 / (k01_x - 0.47) + 1.19
        if backend == 'pyplot':
            fig, ax = plt.subplots()
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            scp = ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='.', c=z, cmap='coolwarm')
            fig.colorbar(scp, ax=ax, label='AGN Fraction')
            ax.plot(k01_x, k01_y, 'k--', lw=.5, label='Kewley et al. 2001 Cutoff')
            ax.legend()
            fig.savefig(fname, dpi=300, bbox_inches='tight')
        elif backend == 'plotly':
            fig = plotly.graph_objects.Figure(
                data=plotly.graph_objects.Scatter(
                    x=x, y=y, mode='markers',
                    error_x=dict(type='data', array=xerr, visible=True),
                    error_y=dict(type='data', array=yerr, visible=True),
                    marker=dict(size=4, color=z, colorscale='bluered', showscale=True),
                    showlegend=False, hovertemplate='(x: %{x:.5f}, y: %{y:.5f}), <br> AGN_frac = %{marker.color:.5f}'
                )
            )
            fig.add_trace(plotly.graph_objects.Scatter(x=k01_x, y=k01_y, line=dict(color='black', width=.5, dash='dash'),
                                                       name='Kewley et al. 2001 Cutoff', showlegend=False))
            fig.update_layout(
                xaxis_title=xl,
                yaxis_title=yl,
                title='Reference point: ({:.5f},{:.5f})'.format(*self.agn_ref_pt)
            )
            fig.update_yaxes(
                range=(np.nanmin(y)-0.05, np.nanmax(y)+0.05),
                constrain='domain'
            )
            fig.update_xaxes(
                range=(np.nanmin(x)-0.05, np.nanmax(x)+0.05),
                constrain='domain'
            )
            fig.write_html(fname)

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
        s += f"Stacked:    \t {True if len(self.stacked_flux) > 0 else False}\n"
        s += f"Binned:     \t {'log_' if self.bin_log else ''}{self.binned}"
        return s
