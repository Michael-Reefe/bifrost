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
import matplotlib.gridspec as gridspec
import plotly.subplots
import plotly.graph_objects

import scipy.optimize
import scipy.integrate
import scipy.interpolate

import astropy.coordinates
import astropy.time
import astropy.io.fits
import astropy.units as u
import astropy.constants as const
import astropy.convolution

import astroquery.irsa_dust
from PyAstronomy.pyasl import fastRotBroad
# import spectres

# Bifrost packages
import bifrost.maths as maths
import bifrost.utils as utils
import bifrost.filters as bfilters


class Spectrum:
    __slots__ = ['wave', 'flux', 'error', 'redshift', 'velocity', 'ra', 'dec', 'ebv', 'name', 'output_path',
                 '_corrected', 'data', '_normalized']

    def __init__(self, wave, flux, error, redshift=None, velocity=None, ra=None, dec=None, ebv=None, name='Generic',
                 output_path=None, **data):
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
            Redshift of the object in dimensionless units.
            ***IMPORTANT***
            If none is provided, it is assumed the given wavelength is already corrected to be in the rest frame of the
            source.
        :param velocity: float
            Radial velocity of the object in km/s.  An alternative that can be used to calculate the redshift if z is
            not known.
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
        self.velocity = velocity
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
        self._normalized = False

        # Other properties that might want to be sorted by
        self.data = data

    def __repr__(self):
        s = '########################################################################################\n'
        s += f"{self.name} Spectrum \n"
        s += '########################################################################################\n'
        corrstr = "(corrected)" if self._corrected else "(uncorrected)"
        s += f"Wavelength range " + corrstr + f":       \t {np.min(self.wave)} - {np.max(self.wave)} angstroms\n"
        s += f"Flux range:                        \t {np.max(np.concatenate(([np.min(self.flux)], [0.0])))} - " \
             f"{np.max(self.flux)} * " \
             f"{'10^-17 erg cm^-2 s^-1 angstrom^-1' if not self.normalized else 'normalized units'}\n"
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
                self.wave = maths.cosmological_redshift(self.wave, self.redshift)
            elif self.velocity:
                self.redshift = maths.calc_redshift_sqrt(self.velocity)
                self.wave = maths.cosmological_redshift(self.wave, self.redshift)
            if self.ebv:
                self.flux = maths.correct_extinction(self.wave, self.flux, self.ebv, r_v)
            self._corrected = True

    def calc_snr(self):
        if 'snr' not in self.data:
            med = np.nanmedian(self.flux)
            self.data['snr'] = 1 / np.mean(self.error[np.where(np.isfinite(self.error))[0]] / med)

    def calc_line_snr(self, wave_range, key):
        good = np.where(np.isfinite(self.error) & (wave_range[0] < self.wave) & (self.wave < wave_range[1]))[0]
        med = np.nanmedian(self.flux[good])
        self.data[key] = 1 / np.mean(self.error[good] / med)

    @property
    def corrected(self):
        return self._corrected

    @corrected.setter
    def corrected(self, value):
        raise ValueError("The 'corrected' property may not be manually set!")

    @corrected.deleter
    def corrected(self):
        raise ValueError("The 'corrected' property may not be deleted!")

    @property
    def normalized(self):
        return self._normalized

    @normalized.setter
    def normalized(self, value):
        raise ValueError("The 'normalized' property may not be manually set!")

    @normalized.deleter
    def normalized(self):
        raise ValueError("The 'normalized' property may not be deleted!")

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
        self.data["agn_frac"] = 1 / d
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
        k_line = 0.61 / (self.data[bpt_x] - 0.47) + 1.19
        agn = self.data[bpt_y] > k_line or self.data[bpt_x] >= 0.47
        self.data["agn_class"] = agn
        return self.data["agn_class"]

    def plot(self, convolve_width=0, emline_color="rebeccapurple", absorp_color="darkgoldenrod", cline_color="cyan",
             overwrite=False, fname=None, backend='plotly', _range=None, ylim=None, normalized=False, title_text=None,
             plot_model=None, shade_reg=None):
        """
        Plot the spectrum.

        :param convolve_width: optional, int
            The width of convolution performed before plotting the spectrum with a Box1DKernel
        :param emline_color: optional, str
            If backend is pyplot, this specifies the color of emission lines plotted.  Default is 'rebeccapurple'.
        :param absorp_color: optional, str
            If backend is pyplot, this specifies the color of absorption lines plotted.  Default is 'darkgoldenrod'.
        :param cline_color: optional, str
            If backend is pyplot, this specifies the color of coronal lines plotted.  Default is 'cyan'.
        :param overwrite: optional, bool
            If true, overwrites the file if it already exists.  Otherwise it is not replotted.  Default is false.
        :param fname: optional, str
            The path and file name to save the plot to.
        :param backend: optional, str
            May be 'pyplot' to use the pyplot module or 'plotly' to use the plotly module for plotting.  Default is
            'plotly'.
        :param _range: optional, tuple
            Limits on the x-data between two wavelengths.
        :param ylim: optional, tuple
            Limits on the y-data between two fluxes.
        :param normalized: optional, bool
            If true, the y axis units are displayed as "normalized".  Otherwise, they are displayed as
            "10^-17 erg cm^-2 s^-1 angstrom^-1". Default is false.
        :param title_text: optional, str
            Text to append to the title of the plot.
        :param plot_model: optional, tuple
            A tuple of two strings corresponding to keys for the data dict to plot x and y data overtop of the spectrum.
        :param shade_reg: optional, list
            A list of tuples indicating region(s) within the plot to shade
        :return None:
        """
        # Make sure corrections have been applied
        self.apply_corrections()
        if not fname:
            fname = os.path.join(self.output_path, self.name.replace(' ', '_') + '.spectrum') + ('.pdf', '.html')[
                backend == 'plotly']
        if os.path.exists(fname) and not overwrite:
            return

        # Convolve the spectrum
        if convolve_width > 0:
            kernel = astropy.convolution.Box1DKernel(convolve_width)
            spectrum = astropy.convolution.convolve(self.flux, kernel)
            error = astropy.convolution.convolve(self.error, kernel)
        else:
            spectrum = self.flux
            error = self.error
        spectrum[spectrum < 0.] = 0.
        error[error < 0.] = 0.
        if _range:
            good = np.where((_range[0] < self.wave) & (self.wave < _range[1]))[0]
            wave = self.wave[good]
            spectrum = spectrum[good]
            error = error[good]
        else:
            wave = self.wave

        if backend == 'pyplot':
            # Plot the spectrum and error
            fig, ax = plt.subplots(figsize=(20, 10))
            linewidth = .5
            linestyle = '--'
            ax.plot(wave, spectrum, '-', color='k', lw=linewidth)
            if plot_model:
                if plot_model[0] in self.data and plot_model[1] in self.data:
                    ax.plot(self.data[plot_model[0]], self.data[plot_model[1]], '-', color='r', lw=linewidth)
                else:
                    print(f"WARNING: {plot_model[0]} or {plot_model[1]} not found in {self.name}'s data!")
            ax.fill_between(wave, spectrum - error, spectrum + error, color='mediumaquamarine', alpha=0.5)

            # Plot emission and absorption lines

            # OVI, Ly-alpha, NV, OI, CII, SiIV, SiIV/OIV, CIV, HeII
            # OIII, AlIII, CIII, CII, NeIV, MgII, [OII]
            # [OII], H-delta, H-gamma, [OIII], H-beta, [OIII], [OIII], [OI], [OI]
            # [NII], H-alpha, [NII], [SII], [SII]
            emlines = np.array(
                [1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                 1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3727.092,
                 3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                 6549.860, 6564.610, 6585.270, 6718.290, 6732.670])
            for line in emlines:
                ax.axvline(line, color=emline_color, lw=linewidth, linestyle=linestyle, alpha=0.5)
            # Ne V, Ne V*, Fe VII, Fe V, Fe V, Ne III (not coronal), Fe V, Fe VII, Fe VI, Fe VII, Fe VI, Fe VII, Fe XIV,
            # Ca V, Fe VI, Ar X, Fe VII, Fe VII*, Fe X, Fe XI
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
            if not normalized:
                ax.set_ylabel(r'$f_\lambda$ ($10^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm{\AA}^{-1}$)', fontsize=fontsize)
            else:
                ax.set_ylabel(r'$f_\lambda$ (normalized)', fontsize=fontsize)
            title = '%s, $z=%.5f$' % (self.name, self.redshift)
            if title_text:
                title += ', ' + title_text
            ax.set_title(title, fontsize=fontsize)
            ax.tick_params(axis='both', labelsize=fontsize - 2)
            if wave.size > 1:
                ax.set_xlim(np.nanmin(wave), np.nanmax(wave))
            if ylim:
                ax.set_ylim(ylim)
            elif wave.size > 1:
                ax.set_ylim(0., np.nanmax(spectrum))
            if shade_reg:
                for sr in shade_reg:
                    ax.axvspan(*sr, color='grey', alpha=0.5)

            fig.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
        elif backend == 'plotly':
            fig = plotly.subplots.make_subplots(rows=1, cols=1)
            linewidth = .5
            fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=spectrum, line=dict(color='black', width=linewidth),
                                                       name='Data', showlegend=False))
            fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=spectrum + error,
                                                       line=dict(color='#60dbbd', width=0),
                                                       fillcolor='rgba(96, 219, 189, 0.6)',
                                                       name='Upper Bound', showlegend=False))
            fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=spectrum - error,
                                                       line=dict(color='#60dbbd', width=0),
                                                       fillcolor='rgba(96, 219, 189, 0.6)',
                                                       fill='tonexty', name='Lower Bound', showlegend=False))
            if plot_model:
                if plot_model[0] in self.data and plot_model[1] in self.data:
                    fig.add_trace(plotly.graph_objects.Scatter(x=self.data[plot_model[0]], y=self.data[plot_model[1]],
                                                               line=dict(color='#e0191c', width=linewidth),
                                                               name='Model', showlegend=False))
                else:
                    print(f"WARNING: {plot_model[0]} or {plot_model[1]} not found in {self.name}'s data!")
            emlines = np.array(
                [1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                 1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                 3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                 6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
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
                fig.add_vline(x=line, line_width=2 * linewidth, line_dash='dot', line_color='#226666',
                              annotation_text=name, annotation_position='top right', annotation_font_size=12)
            for line in abslines:
                fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#d1c779')
            if not normalized:
                y_title = '$f_\\lambda\\ (10^{-17} {\\rm erg} {\\rm s}^{-1} {\\rm cm}^{-2} Å^{-1})$'
            else:
                y_title = '$f_\\lambda\\ ({\\rm normalized})$'
            title = '${\\rm %s}, z=%.5f$' % (self.name, self.redshift)
            if title_text:
                title += ', ' + title_text
            fig.update_layout(
                yaxis_title=y_title,
                xaxis_title='$\\lambda_{\\rm rest}\\ (Å)$',
                title=title,
                hovermode='x',
                template='plotly_white'
            )

            fig.update_layout(
                font_family="Georgia, Times New Roman, Serif",
                # font_color="blue",
                title_font_family="Georgia, Times New Roman, Serif",
                # title_font_color="red",
                # legend_title_font_color="green"
            )
            fig.update_xaxes(title_font_family="Georgia, Times New Roman, Serif")
            fig.update_yaxes(title_font_family="Georgia, Times New Roman, Serif")

            if wave.size > 1:
                _range = (np.nanmin(wave), np.nanmax(wave))
                _yrange = (0, np.nanmax(spectrum) + .3)
            else:
                _range = None
                _yrange = None
            fig.update_xaxes(
                range=_range,
                constrain='domain'
            )
            fig.update_yaxes(
                range=_yrange if not ylim else ylim,
                constrain='domain'
            )
            fig.write_html(fname, include_mathjax="cdn")
            # fig.write_image(fname.replace('.html', '.pdf'), width=1280, height=540)

    def save_pickle(self):
        """
        Save the object contents to a pickle file

        :return None:
        """
        with open(os.path.join(self.output_path, self.name.replace(' ', '_') + '.data.pkl'), 'wb') as handle:
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
        :param save_all_data: bool
            Whether or not to append data within the FITS file to the object's data dictionary
        :return cls: Spectrum
            The Spectrum object created from the fits file.
        """
        # Load the data
        data_dict = {}
        with astropy.io.fits.open(filepath) as hdu:

            specobj = hdu[2].data
            z = specobj['z'][0]
            if name is None:
                name = specobj['SPECOBJID'][0]
                if type(name) is str:
                    name = name.strip()
                else:
                    name = str(name)
            try:
                ra = hdu[0].header['RA']
                dec = hdu[0].header['DEC']
            except KeyError:
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

        # Interpolating over bad pixels
        bad = np.where(~np.isfinite(flux) & ~np.isfinite(error))[0]

        # error[bad] = np.nanmedian(error)

        # Insert additional nans next to bad pixels
        def insert_nans(spec, _bad):
            all_bad = np.unique(np.concatenate((_bad - 1, _bad, _bad + 1)))
            all_bad = np.array([ab for ab in all_bad if 0 < ab < len(spec)])
            try:
                spec[all_bad] = np.nan
                return spec
            except IndexError:
                return spec

        def nan_helper(spec):
            return np.isnan(spec), lambda q: q.nonzero()[0]

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

        return cls(wave, flux, error, redshift=z, ra=ra / 15, dec=dec, ebv=ebv, name=name, **data_dict)

    @classmethod
    def simulated(cls, true_line, wave_range=(-20, 20), rv='random', rv_lim=(-100, 100),
                  vsini='random', vsini_lim=(0, 500), noise_std='random', noise_std_lim=(0.1, 1), a=1, sigma=0.1,
                  name='auto', seed=None):
        """
        Create a simulated spectrum object.

        :param true_line: float
            The wavelength of the emission line to model, in angstroms.
        :param wave_range: tuple
            Range of wavelengths in anstroms to model from the left/right of true_line. Default (-20, 20)
        :param rv: str, float
            The radial velocity, in km/s, of redshift (>0 values are away from the observer). If 'random', draws
            from a uniform distribution limited by rv_lim.
        :param rv_lim: tuple
            Lower and upper boundaries on the uniform distribution that rv draws from if it's 'random'.
            Default (-100, 100)
        :param vsini: str, float
            The rotational velocity, in km/s, of redshift (>0 values are away from the observer). If 'random', draws
            from a uniform distribution limited by vsini_lim.
        :param vsini_lim: tuple
            Lower and upper boundaries on the uniform distribution that vsini draws from if it's 'random'.
            Default (0, 500)
        :param noise_std: str, float
            Standard deviation of the noise to be added to the spectrum.  If 'random', draws from a uniform distribution
            limited by noise_std_lim.
        :param noise_std_lim: tuple
            Lower and upper boundaries on the uniform distribution that noise_std draws from if it's 'random'.
            Default (0.1, 1)
        :param a: float
            Amplitude of the Gaussian model for the emission line BEFORE applying rotational broadening. Default 1.
        :param sigma: float
            Standard deviation of the Gaussian model for the emission line BEFORE applying rotational broadening.
            Default 0.1.
        :param name: str
            Name of the spectrum.
        :param seed: int
            A seed to use in setting random variables, so they may be reproduced.
        :return: Spectrum
            The simulated spectrum object.
        """

        rng = np.random.default_rng(seed)
        # Wavelength grid
        wave = np.arange(true_line + wave_range[0] * 2, true_line + wave_range[1] * 2 + 0.1, 1)

        # Apply redshift first to avoid reinterpolating the flux
        if rv == 'random':
            rv = rng.uniform(*rv_lim)
        z = maths.calc_redshift_sqrt(rv)

        # First, calculate the perfect idealized (unshifted) spectrum
        flux = maths.gaussian(wave, a, true_line, sigma) + 1

        # Then apply rotational broadening
        if vsini == 'random':
            vsini = rng.uniform(*vsini_lim)
        flux = fastRotBroad(wave, flux, 0, vsini=vsini)

        _range = np.where((wave >= true_line + wave_range[0]) & (wave <= true_line + wave_range[1]))[0]
        wave = wave[_range]
        flux = flux[_range]

        if noise_std == 'random':
            noise_std = rng.uniform(*noise_std_lim)
        noise = rng.normal(0, noise_std, wave.size)
        flux += noise
        error = np.array([np.std(flux - 1)] * len(flux))
        snr = 1 / np.std(flux - 1)

        if name == 'auto':
            name = 'Sim_A{:.1f}_s{:d}_{:.3f}_{:.3f}_{:.3f}'.format(a, seed, rv, vsini, noise_std)

        return cls(wave, flux, error, redshift=z, name=name, amp=a, sigma=sigma, snr=snr)

    def to_numpy(self, _slice=None):
        if _slice is None:
            return np.array([(w, f, e) for w, f, e in zip(self.wave, self.flux, self.error)],
                            dtype=[('wave', float), ('flux', float), ('err', float)]).view(np.recarray)
        else:
            return np.array([(w, f, e) for w, f, e in zip(self.wave[_slice], self.flux[_slice], self.error[_slice])],
                            dtype=[('wave', float), ('flux', float), ('err', float)]).view(np.recarray)

    def save_numpy(self, fname=None, compress=True, _slice=None):
        if not self.corrected:
            self.apply_corrections()
        record = self.to_numpy(_slice=_slice)
        save = np.save if not compress else np.savez
        if not fname:
            if compress:
                fname = self.name + '.npz'
            else:
                fname = self.name + '.npy'
        save(fname, record)

    # Arithmetic definitions

    def __add__(self, other):
        assert np.all(self.wave == other.wave), "Cannot add two spectra if their wave arrays do not match!"
        assert self.corrected & other.corrected, "Cannot add two spectra unless both have been properly corrected for" \
                                                 "redshift and extinction!"
        assert (self.normalized & other.normalized) | ((not self.normalized) & (not other.normalized)), \
            "Cannot add two spectra unless both or neither are normalized!"
        result = Spectrum(self.wave, (self.flux + other.flux), np.hypot(self.error, other.error), name='Added spectrum')
        result._normalized = self.normalized & other.normalized
        return result

    def __sub__(self, other):
        assert np.all(self.wave == other.wave), "Cannot subtract two spectra if their wave arrays do not match!"
        assert self.corrected & other.corrected, "Cannot subtract two spectra unless both have been properly corrected for" \
                                                 "redshift and extinction!"
        assert (self.normalized & other.normalized) | ((not self.normalized) & (not other.normalized)), \
            "Cannot subtract two spectra unless both or neither are normalized!"
        result = Spectrum(self.wave, (self.flux - other.flux), np.hypot(self.error, other.error), name='Subtracted spectrum')
        result._normalized = self.normalized & other.normalized
        return result

    def __mul__(self, other):
        assert np.all(self.wave == other.wave), "Cannot multiply two spectra if their wave arrays do not match!"
        assert self.corrected & other.corrected, "Cannot multiply two spectra unless both have been properly corrected for" \
                                                 "redshift and extinction!"
        assert (self.normalized & other.normalized) | ((not self.normalized) & (not other.normalized)), \
            "Cannot multiply two spectra unless both or neither are normalized!"
        result = Spectrum(self.wave, (self.flux * other.flux), np.hypot(other.flux*self.error, self.flux*other.error),
                          name='Multiplied spectrum')
        result._normalized = self.normalized & other.normalized
        return result

    def __truediv__(self, other):
        assert np.all(self.wave == other.wave), "Cannot divide two spectra if their wave arrays do not match!"
        assert self.corrected & other.corrected, "Cannot divide two spectra unless both have been properly corrected for" \
                                                 "redshift and extinction!"
        assert (self.normalized & other.normalized) | ((not self.normalized) & (not other.normalized)), \
            "Cannot divide two spectra unless both or neither are normalized!"
        result = Spectrum(self.wave, (self.flux / other.flux),
                          (self.flux / other.flux)*np.hypot(self.error/self.flux, other.error/other.flux),
                          name='Divided spectrum')
        result._normalized = self.normalized & other.normalized
        return result


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

    def plot_spectra(self, fname_root, spectra='all', _range=None, ylim=None, title_text=None, backend='plotly',
                     plot_model=None, f=None, shade_reg=None):
        """
        Plot a series of spectra from the dictionary.

        :param fname_root: str
            The parent directory to save all plot figures to.
        :param spectra: str, iterable
            Dictionary keys of which spectra to plot. If 'all', all are plotted.  Defaults to all.
        :param _range: optional, tuple
            x-limits on plotted data
        :param ylim: optional, tuple
            y-limits on plotted data
        :param title_text: optional, dict
            Title text to be applied to each plotted spectra. Should be a dictionary with each entry having a key
            of the spectrum name.
        :param backend: str
            'plotly' or 'pyplot'
        :param plot_model: optional, tuple
            A tuple of two strings corresponding to keys for the data dict to plot x and y data overtop of the spectrum.
        :param f: optional, list
            A list containing values corresponding to each plot to be appended to the beginning of the file names.
        :param shade_reg: list
            A list of tuples containing left and right boundaries over which to shade in the plot.
        :return None:
        """
        print('Plotting spectra...')
        fmt = '.html' if backend == 'plotly' else '.pdf'
        if not os.path.exists(fname_root):
            os.makedirs(fname_root)
        if type(spectra) is str:
            if spectra == 'all':
                for i, item in enumerate(tqdm.tqdm(self)):
                    ttl = None if title_text is None else title_text[item]
                    if _range:
                        good = np.where((self[item].wave > _range[0]) & (self[item].wave < _range[1]))[0]
                        if good.size < 10:
                            continue
                    if f is not None:
                        fname = os.path.join(fname_root,
                                             f'{f[i]:.3f}_' + self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    else:
                        fname = os.path.join(fname_root, self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    self[item].plot(fname=fname,
                                    backend=backend, _range=_range, ylim=ylim, title_text=ttl, plot_model=plot_model,
                                    shade_reg=shade_reg)
        else:
            for i, item in enumerate(tqdm.tqdm(self)):
                if item in spectra:
                    if item not in self or item not in title_text:
                        print(f'WARNING: {item} not found in stack!')
                        continue
                    if _range:
                        good = np.where((self[item].wave > _range[0]) & (self[item].wave < _range[1]))[0]
                        if good.size < 10:
                            continue
                    ttl = None if title_text is None else title_text[item]
                    if f is not None:
                        fname = os.path.join(fname_root,
                                             f'{f[i]:.3f}_' + self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    else:
                        fname = os.path.join(fname_root, self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    self[item].plot(fname=fname,
                                    backend=backend, _range=_range, ylim=ylim, title_text=ttl, plot_model=plot_model,
                                    shade_reg=shade_reg)

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
        if t is str or t is np.str_:
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

    def __init__(self, universal_grid=None, stacked_flux=None, stacked_err=None, filters=None, r_v=3.1,
                 gridspace=1, tolerance=500, norm_region=None, wave_criterion='lenient', progress_bar=True):
        """
        An extension of the Spectra class (and by extension, the dictionary) specifically for stacking purposes.

        :param universal_grid: optional, array
            A uniform wavelength grid used for all spectra.  If binned, a list of arrays for each bin.
        :param stacked_flux: optional, array
            An array of the stacked flux. If binned, a list of arrays for each bin.
        :param stacked_err: optional, array
            An array of the stacked flux error. If binned, a list of arrays for each bin.
        :param filters: optional, array
            An array of filters to remove individual spectra that do not satisfy certain conditions.
            Each entry must either be a bifrost Filter object or a string that can be converted into a Filter object.
        :param r_v: float
            Extinction ratio A(V)/E(B-V) to calculate for.  Default = 3.1
        :param gridspace: float
            Spacing of the wavelength grid.  Default = 1
        :param tolerance: float
            Tolerance for throwing out spectra that are > tolerance angstroms apart from others.  Default = 500
        :param norm_region: optional, tuple
            Wavelength bounds to use for the normalization region, with no prominent lines.  Default = None
        :param wave_criterion: str
            One of the following:
                'strict': Completely delete all spectra that do not satisfy wavelength coverage requirements stipulated
                by tolerance and the normalization region.
                'lenient': Do not delete any spectra.  Stack over the entire region, using the full range of spectra
                that have wavelength coverage in different parts of the spectrum.  Will result in a stack with a
                different number of constituent spectra at different wavelength positions.
        :param progress_bar: bool
            If True, shows progress bars for stacking procedures.  Default is False.
        """
        # Fill in instance attributes
        self.r_v = r_v
        self.gridspace = gridspace
        self.tolerance = tolerance
        self.norm_region = norm_region
        if wave_criterion not in ('strict', 'lenient'):
            raise ValueError('wave_criterion must be one of: strict, lenient')
        self.wave_criterion = wave_criterion
        # Filters
        if filters is None:
            filters = []
        else:
            for i in range(len(filters)):
                if type(filters[i]) is str:
                    filters[i] = bfilters.Filter.from_str(filters[i])
        self.filters = filters
        self.progress_bar = progress_bar
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
        self.specnames_f = []
        self.specnames_e = []
        self.nspec_f = []
        self.nspec_e = []
        self.binned = None
        self.binned_spec = None
        self.bin_counts = None
        self.bin_edges = None
        self.bin_log = False
        self.agn_ref_pt = None
        super().__init__()

    def calc_norm_region(self, wave_grid):
        """
        Calculate the optimal region to perform normalization. Finds the largest span of wavelengths between
        absportion lines that is also covered by all the spectra in the dictionary.  Fails if no such region can
        be found.  The region is set to the instance attribute self.norm_region.

        :param wave_grid: iterable
            A uniform wave grid used for all spectra.
        :return nr0, nr1: tuple
            The left and right edges of the normalization region.
        """
        if self.wave_criterion == 'strict':
            emlines = np.array(
                [1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                 1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3346.790, 3426.850, 3727.092,
                 3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                 6374.510, 6549.860, 6564.610, 6585.270, 6718.290, 6732.670, 7891.800])
            abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
            lines = np.concatenate((emlines, abslines))
            lines.sort()
            diffs = np.diff(lines)
            for _ in range(len(diffs)):
                imax = np.nanargmax(diffs)
                nr0, nr1 = lines[imax], lines[imax + 1]
                if wave_grid[0] < nr0 < wave_grid[-1] and wave_grid[0] < nr1 < wave_grid[-1]:
                    self.norm_region = (nr0, nr1)
                    return nr0, nr1
                else:
                    diffs[imax] = np.nan
            print("WARNING: An ideal normalization region could not be found!  Using the entire range.")
            self.norm_region = (wave_grid[0], wave_grid[-1])
            return wave_grid[0], wave_grid[-1]
        elif self.wave_criterion == 'lenient':
            print("WARNING: wave_criterion is lenient, so a constrained normalization region cannot be calculated."
                  " Using the ENTIRE wavelength range as the normalization region.")
            return wave_grid[0], wave_grid[-1]
        else:
            raise ValueError('invalid value for self.wave_criterion!')

    def filter_spectra(self):
        """
        Go through each filter and decide if each spectrum fits its criteria. If not, the specttrum is masked out.

        :return None:
        """
        aliases = {'z': 'redshift'}
        for filt in self.filters:
            if filt.attribute in aliases:
                att = aliases[filt.attribute]
            else:
                att = filt.attribute
            removals = []
            for ispec in self:
                if not (filt.lower_bound < getattr(self[ispec], att) < filt.upper_bound):
                    print(f"WARNING: Removing spectrum {self.get_spec_index(ispec) + 1}: {ispec} "
                          f"since it does not fulfill the criteria: {filt}")
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
            _id = data[i][bin_quantity]
            if log:
                _id = np.log10(_id)
            if np.isnan(_id):
                print(f"WARNING: bin_quantity is {_id} in {ispec} data!  Ignoring this spectrum")
                continue
            included = np.append(included, ispec)
            unbinned = np.append(unbinned, _id)

        # Perform the binning
        minbin = np.nanmin(unbinned)
        maxbin = np.nanmax(unbinned)
        if bin_size:
            nbins = -int(-(maxbin - minbin) // bin_size)
        elif nbins:
            bin_size = (maxbin - minbin) / nbins
        if round_bins:
            rating = 1 / round_bins
            minbin = np.floor(minbin * rating) / rating
            maxbin = np.ceil(maxbin * rating) / rating
            bin_size = np.round(bin_size * rating) / rating
            if bin_size == 0:
                bin_size = round_bins
            nbins = -int(-(maxbin - minbin) // bin_size)

        binned_spec = {i: np.array([], dtype=np.str) for i in range(nbins)}
        bin_counts = np.zeros(nbins)

        bin_edges = minbin + np.arange(0, nbins + 1, 1) * bin_size
        bin_midpts = minbin + (np.arange(0, nbins, 1) + 0.5) * bin_size
        for i in range(len(included)):
            indx = int((unbinned[i] - minbin) / bin_size)
            if indx == len(binned_spec):
                indx -= 1
            binned_spec[indx] = np.append(binned_spec[indx], included[i])
            bin_counts[indx] += 1

        if midpoints:
            return binned_spec, bin_counts, bin_midpts
        return binned_spec, bin_counts, bin_edges

    def histogram_3d(self, fname_base, bin_quantities, logs, nbins=None, bin_size=None, round_bins=None, labels=None,
                     backend='plotly', colormap=None):
        """
        Make a 3D histogram of the data using 3 quanitites.

        :param fname_base: str
            File name without the extension.
        :param bin_quantities: iterable
            List of the names of quantities to bin by.
        :param logs: iterable
            List of booleans on whether to take the log10 of each bin quantity.
        :param nbins: iterable
            List of the number of bins to use for each quantity.
        :param bin_size: iterable
            List of the size of each bin to use for each quantity.
        :param round_bins: iterable
            List of booleans whether to round bins nicely for each quantity.
        :param labels: iterable
            List of labels for each bin.
        :param backend: str
            'plotly' or 'pyplot'
        :param colormap: str or matplotlib colormap object
            The colormap to use for the 3rd bin quantity.
        :return:
        """

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
        for x, y in itertools.product(np.arange(nbx), np.arange(nby)):
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
            fig, ax = plt.subplots(figsize=(nbx / nby * 7.5 + 3.5, 7.5))
            mesh = ax.pcolormesh(edgex, edgey, z_array, shading='flat', cmap=colormap)

            if not labels:
                xlabel = binx if not logx else '$\\log($' + binx + '$)$'
                ylabel = biny if not logy else '$\\log($' + biny + '$)$'
                zlabel = binz if not logz else '$\\log($' + binz + '$)$'
            else:
                xlabel, ylabel, zlabel = labels
            fig.colorbar(mesh, ax=ax, label=zlabel)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            for x, y in itertools.product(np.arange(nbx), np.arange(nby)):
                if n_array[y, x] > 0:
                    ax.text(edgex[x] + (edgex[x + 1] - edgex[x]) / 2, edgey[y] + (edgey[y + 1] - edgey[y]) / 2,
                            str(n_array[y, x]),
                            fontsize=7, horizontalalignment='center', verticalalignment='center', color='white')

            ax.set_xticks(edgex[::3])
            ax.set_yticks(edgey)
            fig.savefig(fname_base + '.pdf', dpi=300, bbox_inches='tight')
            plt.close()
        elif backend == 'plotly':
            if not colormap:
                colormap = 'plasma'
            if not labels:
                xlabel = binx if not logx else '$\\log(' + binx + ')$'
                ylabel = biny if not logy else '$\\log(' + biny + ')$'
                zlabel = binz if not logz else '$\\log(' + binz + ')$'
            else:
                xlabel, ylabel, zlabel = labels
            fig = plotly.graph_objects.Figure(
                data=plotly.graph_objects.Heatmap(x=edgex, y=edgey, z=z_array,
                                                  colorbar=dict(title=zlabel), colorscale=colormap)
            )
            for x, y in itertools.product(np.arange(nbx), np.arange(nby)):
                if n_array[y, x] > 0:
                    fig.add_annotation(text=str(n_array[y, x]), xref="x", yref="y",
                                       x=edgex[x] + (edgex[x + 1] - edgex[x]) / 2,
                                       y=edgey[y] + (edgey[y + 1] - edgey[y]) / 2, showarrow=False,
                                       font=dict(size=7, color="white"))
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

            fig.update_layout(
                template='plotly_white',
                font_family="Georgia, Times New Roman, Serif",
                # font_color="blue",
                title_font_family="Georgia, Times New Roman, Serif",
                # title_font_color="red",
                # legend_title_font_color="green"
            )
            fig.update_xaxes(title_font_family="Georgia, Times New Roman, Serif")
            fig.update_yaxes(title_font_family="Georgia, Times New Roman, Serif")

            # fig.update_layout(
            # paper_bgcolor='rgba(0,0,0,0)',
            # plot_bgcolor='rgba(0,0,0,0)'
            # )
            fig.write_html(fname_base + '.html', include_mathjax="cdn")
            fig.write_image(fname_base + '.pdf')

    def kewley_agn_class(self, bpt_1, bpt_2):
        for i, ispec in enumerate(self):
            self[ispec].k01_agn_class(bpt_1, bpt_2)

    # Allow the class to be called as a way to perform the stacking
    @utils.timer(name='Stack Procedure')
    def __call__(self, bin_name=None, nbins=None, bin_size=None, log=False, round_bins=None, auto_norm_region=True,
                 bpt_1=None, bpt_2=None, hbin_target=3, stack_all_agns=False):
        """
        The main procedure for stacking spectra.  Performs all necessary steps at once:
            1. Convert each spectra to their rest-frame wavelengths using their redshifts.
            2. Correct each spectra's flux for galactic extinction.
            3. Find the optimal, universal wavelength grid that is in a region shared by all spectra and with uniform
               spacing.
            4. Resample each spectrum in the dictionary onto the universal grid, while conserving flux and flux error.
            5. Normalize each spectrum to the median value in the normalization region.
            6. Coadd each spectrum together using 1/error^2 as the weights at each pixel value.
            7. Coadd the errors for each spectrum together again using 1/error^2 as the weights at each pixel value.

        :param bin_name: optional, str
            The name of a quantity to bin the data by before stacking each bin. Must exist in each Spectrum's data
            dictionary.
        :param nbins: optional, int
            The number of bins to use.
            If 'bin' is specified, then one of 'nbins' or 'bin_size' must also be specified.
        :param bin_size: optional, float
            The size of each bin.
            If 'bin' is specified, then one of 'nbins' or 'bin_size' must also be specified.
        :param log: optional, boolean
            If True, takes the log10 of the bin quantity BEFORE binning. 'nbins' and 'bin_size' should be specified
            according to the log10 of the bin quantity.
        :param round_bins: optional, float
            If specified, rounds the bin edges / bin sizes to be multiples of this quantity.
        :param auto_norm_region: optional, boolean
            If True, automatically calculates the normalization region for each stacked bin. Otherwise, uses the default
            value specified by the norm_region instance attribute.
        :param bpt_1: optional, str
            The name of the BPT x-value to be used if 'stack_all_agns' is True. Must exist in each Spectrum's data
            dictionary.
        :param bpt_2: optional, str
            The name of the BPT y-value to be used if 'stack_all_agns' is True. Must exist in each Spectrum's data
            dictionary.
        :param hbin_target: int
            The targeted minimum number of galaxies to include in the highest bin, if stacking by AGN distance, where
            the distance is in relation to an arbitrary reference point that we have the freedom to choose.
            Default is 3.
        :param stack_all_agns: boolean
            If True, uses the Kewley et al. 2001 AGN criteria as a cutoff and stacks ALL galaxies that satisfy this
            criteria together. Mutually incompatible with 'bin' and its related arguments.

        :return None:
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
                nr0, nr1 = self.calc_norm_region(wave_grid_b)
            else:
                nr0, nr1 = self.norm_region
            self.normalize(wave_grid_b, (nr0, nr1), spectra)
            if self.wave_criterion == 'strict':
                wave_grid_b, flux_b, err_b = self.coadd(wave_grid_b, spectra)
            elif self.wave_criterion == 'lenient':
                wave_grid_b, flux_b, err_b, specnames_fb, specnames_eb, nspec_fb, nspec_eb = \
                    self.coadd(wave_grid_b, spectra, save_specnames=True)
                self.specnames_f.append(specnames_fb)
                self.specnames_e.append(specnames_eb)
                self.nspec_f.append(nspec_fb)
                self.nspec_e.append(nspec_eb)
            else:
                raise ValueError('invalid value for self.wave_criterion!')
            self.universal_grid.append(wave_grid_b)
            self.stacked_flux.append(flux_b)
            self.stacked_err.append(err_b)

        elif bin_name:
            if bin_name == 'agn_frac':

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
                    counts = len(np.where(agn_fracs >= 1 - bin_size)[0])

                    print('Current number in highest bin: %03d' % counts, end='\r', flush=True)
                    if counts > target_num:
                        return 0
                    else:
                        return -1 * counts

                # USE POWELL METHOD
                print('Optimizing AGN fraction reference point...')
                min_res = scipy.optimize.minimize(num_in_highest_bin, 0,
                                                  args=(hbin_target, bin_size, nbins, round_bins), method='Powell')
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

            binned_spectra, bin_counts, bin_edges = self.bin_spectra(bin_name, bin_size=bin_size, nbins=nbins, log=log,
                                                                     round_bins=round_bins)
            nn = len(binned_spectra)
            for i, b in enumerate(binned_spectra):
                print(f'BIN {i + 1} OF {nn}...')
                spectra = binned_spectra[b]
                if len(spectra) == 0:
                    self.universal_grid.append(None)
                    self.stacked_flux.append(None)
                    self.stacked_err.append(None)
                    continue
                wave_grid_b, spectra = self.uniform_wave_grid(spectra)
                self.resample(wave_grid_b, spectra)
                if auto_norm_region:
                    nr0, nr1 = self.calc_norm_region(wave_grid_b)
                else:
                    nr0, nr1 = self.norm_region
                self.normalize(wave_grid_b, (nr0, nr1), spectra)
                if self.wave_criterion == 'strict':
                    wave_grid_b, flux_b, err_b = self.coadd(wave_grid_b, spectra)
                elif self.wave_criterion == 'lenient':
                    wave_grid_b, flux_b, err_b, specnames_fb, specnames_eb, nspec_fb, nspec_eb = \
                        self.coadd(wave_grid_b, spectra, save_specnames=True)
                    self.specnames_f.append(specnames_fb)
                    self.specnames_e.append(specnames_eb)
                    self.nspec_f.append(nspec_fb)
                    self.nspec_e.append(nspec_eb)
                else:
                    raise ValueError('invalid value for self.wave_criterion!')
                self.universal_grid.append(wave_grid_b)
                self.stacked_flux.append(flux_b)
                self.stacked_err.append(err_b)
            self.binned = bin_name
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
            if self.wave_criterion == 'strict':
                wave_grid, flux, err = self.coadd(wave_grid)
            elif self.wave_criterion == 'lenient':
                wave_grid, flux, err, specnames_f, specnames_e, nspec_f, nspec_e = \
                    self.coadd(wave_grid, save_specnames=True)
                self.specnames_f.append(specnames_f)
                self.specnames_e.append(specnames_e)
                self.nspec_f.append(nspec_f)
                self.nspec_e.append(nspec_e)
            else:
                raise ValueError('invalid value for self.wave_criterion!')
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
        binned_indices = np.array([self.get_spec_index(name) for name in all_names], dtype=int)
        wave = self.to_numpy('wave')['wave'][binned_indices]
        if self.wave_criterion == 'strict':
            wmin = None
            wmax = None
            removed_names = np.array([], dtype=int)
            for i, wi in enumerate(wave):
                remove = False
                if self.norm_region:
                    if np.where((self.norm_region[0] - 5 < wi) & (self.norm_region[0] + 5 > wi))[0].size == 0 or \
                            np.where((self.norm_region[1] - 5 < wi) & (self.norm_region[1] + 5 > wi))[0].size == 0:
                        remove = True
                if not remove:
                    if not wmin or not wmax:
                        wmin = np.nanmin(wi)
                        wmax = np.nanmax(wi)
                        continue
                    imin = np.nanmin(wi)
                    if imin > wmin:
                        if np.abs(imin - wmin) > self.tolerance:
                            remove = True
                        else:
                            wmin = imin
                    imax = np.nanmax(wi)
                    if imax < wmax:
                        if np.abs(imax - wmax) > self.tolerance:
                            remove = True
                        else:
                            wmax = imax
                if remove:
                    print(
                        f"WARNING: Removing spectrum {i + 1}: {all_names[i]} due to insufficient wavelength coverage.")
                    del self[all_names[i]]
                    removed_names = np.append(removed_names, i)
            all_names = np.delete(all_names, removed_names)
        elif self.wave_criterion == 'lenient':
            wmin = 1e100
            wmax = -1e100
            for i, wi in enumerate(wave):
                imin = np.nanmin(wi)
                imax = np.nanmax(wi)
                if imin < wmin:
                    wmin = imin
                if imax > wmax:
                    wmax = imax
        else:
            raise ValueError('invalid value for self.wave_criterion!')

        wave_grid = np.arange(int(wmin), int(wmax) + self.gridspace, self.gridspace)
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
        range_ = tqdm.tqdm(ss) if self.progress_bar else ss
        for ispec in range_:
            self[ispec].flux, self[ispec].error = \
                maths.spectres(wave_grid, self[ispec].wave, self[ispec].flux,
                               self[ispec].error, fill=np.nan,
                               verbose=False if self.wave_criterion == 'lenient' else True)
            self[ispec].wave = wave_grid
        print('Done.')

    def normalize(self, wave_grid, norm_region, binned_spec=None):
        """
        Normalize all spectra by the median of the normalization region.

        :param wave_grid: np.ndarray
            The grid of wavelengths to resample to.
        :param norm_region: tuple
            The left and right edges of wavelength to normalize by.
        :param binned_spec: iterable
            A list of spectra names to use.  If None, all are used.
        :return None:
        """
        print('Normalizing spectra...')
        # Use the first spectra's wave since by this point they should all be equal anyways,
        # to calculate the region to fit
        reg = np.where((norm_region[0] < wave_grid) & (wave_grid < norm_region[1]))[0]
        ss = binned_spec if binned_spec is not None else [s for s in self]
        range_ = tqdm.tqdm(ss) if self.progress_bar else ss
        for ispec in range_:
            self[ispec].flux, self[ispec].error = self._norm(self[ispec].flux, self[ispec].error, reg)
            self[ispec]._normalized = True
        print('Done.')

    @staticmethod
    @njit
    def _norm(data, error, region):
        med = np.nanmedian(data[region])
        data /= med
        error /= med
        return data, error

    # def _renorm_stack(self, norm_region):
    #     """
    #     Renormalize the stacked spectra and all individual spectra within each stack to a new normalization region.
    #
    #     :param norm_region: tuple
    #         The new lower and upper limits for the region to normalize by.
    #     :return None:
    #     """
    #     for i in range(len(self.stacked_flux)):
    #         reg = np.where((norm_region[0] < self.universal_grid[i]) & (self.universal_grid[i] < norm_region[1]))[0]
    #         med = np.nanmedian(self.stacked_flux[i][reg])
    #         self.stacked_flux[i] /= med
    #         if self.binned_spec:
    #             ss = self.binned_spec[i]
    #         else:
    #             ss = self
    #         for ispec in ss:
    #             medi = np.nanmedian(self[ispec].flux[reg])
    #             self[ispec].flux /= medi

    def coadd(self, wave_grid, binned_spec=None, save_specnames=False):
        """
        Coadd all spectra together into a single, stacked spectrum, using 1/sigma**2 as the weights.

        :param wave_grid: np.ndarray
            The grid of wavelengths to resample to.
        :param binned_spec: iterable
            A list of spectra names to use.  If None, all are used.
        :param save_specnames: bool
            Whether or not to save the names and numbers of spectra that are coadded at each wavelength point.
        :return None:
        """
        stacked_flux = np.zeros_like(wave_grid, dtype=np.float64)
        stacked_err = np.zeros_like(wave_grid, dtype=np.float64)

        print('Coadding spectra...')
        ss = binned_spec if binned_spec is not None else [s for s in self]
        specnames_f = np.ndarray(wave_grid.size, dtype=object)
        specnames_e = np.ndarray(wave_grid.size, dtype=object)
        nspec_f = np.zeros_like(wave_grid)
        nspec_e = np.zeros_like(wave_grid)
        range_ = tqdm.trange(len(wave_grid)) if self.progress_bar else range(len(wave_grid))
        for i in range_:
            flux_i = np.array([self[name].flux[i] for name in ss])
            err_i = np.array([self[name].error[i] for name in ss])
            if save_specnames:
                specnames_f[i] = np.array([name for name in ss if np.isfinite(self[name].flux[i])])
                specnames_e[i] = np.array([name for name in ss if np.isfinite(self[name].error[i])])
                nspec_f[i] = specnames_f[i].size
                nspec_e[i] = specnames_e[i].size
            if len(ss) > 1:
                stacked_flux[i], stacked_err[i] = self._coadd_flux_err(flux_i, err_i)
            else:
                stacked_flux[i], stacked_err[i] = flux_i, err_i

        good = np.where(np.isfinite(stacked_flux) & np.isfinite(stacked_err))[0]
        print('Done.')
        if save_specnames:
            return wave_grid[good], stacked_flux[good], stacked_err[good], specnames_f[good], specnames_e[good], \
                   nspec_f[good], nspec_e[good]
        else:
            return wave_grid[good], stacked_flux[good], stacked_err[good]

    @staticmethod
    @njit
    def _coadd_flux_err(flux, error):
        weights = 1 / error ** 2
        M = len(np.where(weights > 0)[0])
        if np.isnan(flux).all() or np.isnan(error).all() or M <= 1:
            return np.nan, np.nan
        stacked_flux = np.nansum(flux * weights) / np.nansum(weights)
        stacked_err = np.sqrt(
            np.nansum((flux - stacked_flux) ** 2 * weights) / ((M - 1) / M * np.nansum(weights))
        )
        # stacked_err = np.sqrt(1/np.nansum(weights))
        return stacked_flux, stacked_err

    @utils.timer(name='Line Flux Integration')
    def calc_line_flux_ratios(self, line, dw=5, tag='', save=False, conf=None, path=''):
        """
        Calculate the integrated flux ratios of each spectra compared to the stacked spectrum.

        :param line: float, int
            The center wavelength at which to integrate (angstroms).
        :param dw: float, int
            The distance to the left/right of the center wavelength to integrate (angstroms).
        :param tag: string
            An optional tag string to add to the end of saved file names.
        :param save: boolean
            If True, saves the line flux ratios as a json file.
        :param conf: str
            Key for a confidence parameter in each spectrum's dictionary to compare to the line flux ratios.
        :param path: str
            Output path for the json file if 'save' is True.
        :return out: dict
            Dictionary of keys: spectra names, and values: tuple(integrated flux, error) / stacked spectrum integrated
            flux.
        """
        out = {0: {}}
        confs = {0: {}}
        self.correct_spectra()
        # if len(self.universal_grid) == 0:
        #     raise ValueError("Stacked spectrum has not yet been generated!")
        # self._renorm_stack((line-norm_dw, line+norm_dw))

        range_ = tqdm.trange(len(self)) if self.progress_bar else range(len(self))
        _wr = 30
        _wl = 30
        if 5275 < line < 5277:
            _wr += 55
        elif 5302 < line < 5304:
            _wl += 30
            _wr += 30
        elif 5308 < line < 5310:
            _wl += 30
            _wr += 30
        elif 5334 < line < 5336:
            _wl += 55
        elif 5719 < line < 5721:
            _wr += 30
        for i in range_:
            # Define wavelength windows
            window_center = (self[i].wave > line - dw) & (self[i].wave < line + dw)
            window_left = (self[i].wave > line - dw - _wl) & (self[i].wave < line + dw - _wl)
            window_right = (self[i].wave > line - dw + _wr) & (self[i].wave < line + dw + _wr)

            if len(window_center) < int(2 * dw) or len(window_left) < int(2 * dw) or len(window_right) < int(2 * dw):
                print(f"WARNING: {self[i].name} spectrum does not have sufficient wavelength coverage in the"
                      f" integration region.")
                continue
            bad = ~np.isfinite(self[i].flux) | ~np.isfinite(self[i].wave)
            if len(np.where(bad & window_center)[0]) >= 3 or len(np.where(bad & window_left)[0]) >= 3 or len(
                    np.where(bad & window_right)[0]) >= 3:
                print(f"WARNING: {self[i].name} spectrum does not have sufficient wavelength coverage in the "
                      f"integration region.")
                continue
            # window_center = np.where(window_center)[0]
            window_left = np.where(window_left)[0]
            window_right = np.where(window_right)[0]

            # Normalize stack in the region of interest
            window_full = np.where((self[i].wave > line - dw - _wl) & (self[i].wave < line + dw + _wr))[0]
            flux_norm, err_norm = self._norm(self[i].flux, self[i].error, window_full)

            # Calculate a linear trend
            mean_left = np.nanmean(flux_norm[window_left])
            mean_right = np.nanmean(flux_norm[window_right])
            slope = (mean_right - mean_left) / (_wl + _wr)
            intercept = mean_left - slope * (line - _wl)

            full_wave = self[i].wave[window_full]
            y = slope * full_wave + intercept

            # Detrend with the line
            full_flux = flux_norm[window_full] - y

            # Calculate mean / RMS
            window_lr = np.where(((full_wave > line - dw - _wl) & (full_wave < line + dw - _wl)) | (
                    (full_wave > line - dw + _wr) & (full_wave < line + dw + _wr)))[0]
            window_center = np.where((full_wave > line - dw) & (full_wave < line + dw))[0]
            rms = np.sqrt(np.mean(full_flux[window_lr] ** 2))

            out[0][self[i].name] = (np.mean(full_flux[window_center]) / rms).astype(np.float64)
            if conf:
                confs[0][self[i].name] = self[i].data[conf]

        # OLD INTEGRATION METHOD, DEPRECATED, COMPARING WITH STACK:
        # for i in range(len(self.universal_grid)):
        #     print(f"BIN {i+1} of {len(self.universal_grid)}...")
        #     cspline_stack = scipy.interpolate.CubicSpline(self.universal_grid[i], self.stacked_flux[i],
        #                                                   extrapolate=False)
        #
        #     baseline, err = scipy.integrate.quad(cspline_stack.__call__, line-dw, line+dw)
        #     out[i] = {"stack": (baseline, err)}
        #     confs[i] = {}
        #     if self.binned_spec:
        #         ss = self.binned_spec[i]
        #     else:
        #         ss = self
        #     print('Calculating relative line flux ratios...')
        #     range_ = tqdm.tqdm(ss) if self.progress_bar else ss
        #     for ispec in range_:
        #         good = np.isfinite(self[ispec].wave) & np.isfinite(self[ispec].flux) & np.isfinite(self[ispec].error)
        #         region = (line-dw < self[ispec].wave) & (self[ispec].wave < line+dw)
        #         bad = ~np.isfinite(self[ispec].error) | ~np.isfinite(self[ispec].flux) |
        #               ~np.isfinite(self[ispec].wave)
        #         if len(np.where(bad & region)[0]) >= 3:
        #             print(f"WARNING: {ispec} spectrum has undefined datapoints in the line region!  "
        #                   f"Cannot calculate relative line flux.")
        #             continue
        #         good = np.where(good)[0]
        #         csplinei = scipy.interpolate.CubicSpline(self[ispec].wave[good], self[ispec].flux[good],
        #                                                  extrapolate=False)
        #         intflux, erri = scipy.integrate.quad(csplinei.__call__, line-dw, line+dw)
        #         out[i][ispec] = (intflux/baseline, err/baseline)
        #         if conf:
        #             confs[i][ispec] = self[ispec].data[conf]
        #     print('Done.')

        if save:
            serialized = json.dumps(out, indent=4)
            with open(path + os.sep + 'line_flux_ratios_' + str(line) + '_' + tag + '.json', 'w') as handle:
                handle.write(serialized)
        if conf:
            return _wl, _wr, out, confs
        else:
            return _wl, _wr, out

    def plot_stacked(self, fname_base, emline_color="rebeccapurple", absorp_color="darkgoldenrod", cline_color="cyan",
                     cline_labels='all', backend='plotly'):
        """
        Plot the stacked spectrum.

        :param fname_base: str
            The path and file name to save the figure to.
        :param emline_color: str
            If backend is 'pyplot', this specifies the color of the plotted emission lines.  Default is 'rebeccapurple'.
        :param absorp_color: str
            If backend is 'pyplot', this specifies the color of the plotted absorption lines. Default is
            'darkgoldenrod'.
        :param cline_color: str
            If backend is 'pyplot', this specifies the color of the plotted coronal lines. Default is 'cyan'.
        :param cline_labels: str, list
            Which coronal line labels to include.  If 'all', include all labels.  Applies only if backend is 'plotly'.
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
            format1 = '_' + str(bin_num) + format0
            fname = fname_base + format1
            wave = self.universal_grid[bin_num]
            flux = self.stacked_flux[bin_num]
            err = self.stacked_err[bin_num]
            if wave is None or flux is None or err is None:
                continue

            # Plot the spectrum and error
            if backend == 'pyplot':
                if self.wave_criterion == 'strict':
                    fig, ax = plt.subplots(figsize=(40, 10))
                    ax.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)', fontsize=20)
                elif self.wave_criterion == 'lenient':
                    gs = gridspec.GridSpec(nrows=20, ncols=20)
                    fig = plt.figure(constrained_layout=True)
                    ax = fig.add_subplot(gs[0:18, :18])
                    ax.tick_params(
                        axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)  # labels along the bottom edge are off
                    ax2 = fig.add_subplot(gs[19, :18], sharex=ax)
                    ax2.set_xlabel(r'$\lambda_{\rm{rest}}$ ($\rm{\AA}$)', fontsize=20)
                else:
                    raise ValueError('invalid value for self.wave_criterion!')
                linewidth = .5
                linestyle = '--'
                ax.plot(wave, flux, '-', color='k', lw=linewidth)
                ax.fill_between(wave, flux - err, flux + err, color='mediumaquamarine', alpha=0.5)
                if self.wave_criterion == 'lenient':
                    extent = [wave[0] - (wave[1] - wave[0]) / 2., wave[-1] + (wave[1] - wave[0]) / 2., 0, 1]
                    nspec = ax2.imshow(self.nspec_f[bin_num][np.newaxis, :], aspect='auto', cmap='plasma',
                                       extent=extent)
                    fig.colorbar(nspec, cax=fig.add_subplot(gs[:, 19]), label='Number of Galaxies')

                # Plot emission and absorption lines

                # OVI, Ly-alpha, NV, OI, CII, SiIV, SiIV/OIV, CIV, HeII
                # OIII, AlIII, CIII, CII, NeIV, MgII, NeV, NeVI, [OII]
                # [OII], H-delta, H-gamma, [OIII], H-beta, [OIII], [OIII], [OI], [OI]
                # [FeX], [NII], H-alpha, [NII], [SII], [SII], [FeXI]
                emlines = np.array(
                    [1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                     1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3727.092,
                     3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                     6549.860, 6564.610, 6585.270, 6718.290, 6732.670])

                for line in emlines:
                    ax.axvline(line, color=emline_color, lw=linewidth, linestyle=linestyle, alpha=0.5)

                # Ne V, Ne V*, Fe VII, Fe V, Fe V, Ne III (not coronal), Fe V, Fe VII, Fe VI, Fe VII, Fe VI, Fe VII,
                # Fe XIV, Ca V, Fe VI, Ar X, Fe VII, Fe VII*, Fe X, Fe XI
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
                ax.set_ylabel(r'$f_\lambda$ (normalized)', fontsize=fontsize)
                ax.set_title('%s' % 'Stacked Spectrum', fontsize=fontsize)
                ax.tick_params(axis='both', labelsize=fontsize - 2)
                ax.set_xlim(np.nanmin(wave), np.nanmax(wave))
                ax.set_ylim(0., np.nanmax(flux) + .3)

                fig.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close()
            elif backend == 'plotly':
                fig = plotly.subplots.make_subplots(rows=1, cols=1)
                linewidth = .5
                if self.wave_criterion == 'lenient':
                    fig.add_trace(
                        plotly.graph_objects.Heatmap(x=wave, y=np.array([0] * len(wave)), z=self.nspec_f[bin_num],
                                                     colorbar=dict(title='Number of spectra'),
                                                     name='Number of Spectra', showlegend=False))
                    text = [str(self.nspec_f[bin_num][i]) for i in range(len(self.nspec_f[bin_num]))]
                elif self.wave_criterion == 'strict':
                    text = [str(len(self)) for _ in range(len(self.stacked_flux[bin_num]))]
                else:
                    raise ValueError('invalid value for self.wave_criterion!')
                fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=flux, line=dict(color='black', width=linewidth),
                                                           name='Data', showlegend=False, text=text,
                                                           hovertemplate='%{y} <b>Number of Spectra:</b> %{text}'))
                fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=flux + err,
                                                           line=dict(color='#60dbbd', width=0),
                                                           fillcolor='rgba(96, 219, 189, 0.6)',
                                                           name='Upper Bound', showlegend=False, hovertemplate='%{y}'))
                fig.add_trace(plotly.graph_objects.Scatter(x=wave, y=flux - err,
                                                           line=dict(color='#60dbbd', width=0),
                                                           fillcolor='rgba(96, 219, 189, 0.6)',
                                                           fill='tonexty', name='Lower Bound', showlegend=False,
                                                           hovertemplate='%{y}'))

                emlines = np.array(
                    [1033.820, 1215.240, 1240.810, 1305.530, 1335.310, 1397.610, 1399.800, 1549.480, 1640.400,
                     1665.850, 1857.400, 1908.734, 2326.000, 2439.500, 2799.117, 3727.092,
                     3729.875, 4102.890, 4341.680, 4364.436, 4862.680, 4960.295, 5008.240, 6300.304, 6363.776,
                     6549.860, 6564.610, 6585.270, 6718.290, 6732.670])
                clines = np.array(
                    [3346.790, 3426.850, 3759, 3839, 3891, 3970, 4181, 4893, 5146, 5159, 5176, 5276, 5303, 5309, 5335,
                     5533, 5720, 6087, 6374.510, 7891.800])
                cline_names = np.array(
                    ['[Ne V]', '[Ne V]*', '[Fe VII]', '[Fe V]', '[Fe V]', '[Ne III]', '[Fe V]', '[Fe VII]', '[Fe VI]',
                     '[Fe VII]', '[Fe VI]', '[Fe VII]', '[Fe XIV]', '[Ca V]', '[Fe VI]', '[Ar X]', '[Fe VII]',
                     '[Fe VII]*',
                     '[Fe X]', '[Fe XI]'], dtype=str
                )
                abslines = np.array([3934.777, 3969.588, 5176.700, 5895.600, 8500.3600, 8544.440, 8664.520])
                for line in emlines:
                    fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#663399')
                for line, name in zip(clines, cline_names):
                    if cline_labels == 'all' or (type(cline_labels) is list and name in cline_labels):
                        fig.add_vline(x=line, line_width=2 * linewidth, line_dash='dot', line_color='#226666',
                                      annotation_text=name, annotation_position='top right', annotation_font_size=12)
                for line in abslines:
                    fig.add_vline(x=line, line_width=linewidth, line_dash='dash', line_color='#d1c779')
                title = 'Stacked Spectra'
                fig.update_layout(
                    yaxis_title='$f_{\\lambda}\\ ({\\rm normalized})$',
                    xaxis_title='$\\lambda_{\\rm rest}\\ (Å)$',
                    title=title,
                    hovermode='x unified',
                    template='plotly_white'
                )

                fig.update_layout(
                    font_family="Georgia, Times New Roman, Serif",
                    # font_color="blue",
                    title_font_family="Georgia, Times New Roman, Serif",
                    # title_font_color="red",
                    # legend_title_font_color="green"
                )
                fig.update_xaxes(title_font_family="Georgia, Times New Roman, Serif")
                fig.update_yaxes(title_font_family="Georgia, Times New Roman, Serif")

                fig.update_xaxes(
                    range=(np.nanmin(wave), np.nanmax(wave)),
                    constrain='domain'
                )
                fig.update_yaxes(
                    range=(0, np.nanmax(flux) + err[np.nanargmax(flux)] + .3),
                    constrain='domain'
                )
                fig.write_html(fname, include_mathjax="cdn")
                fig.write_image(fname.replace('.html', '.pdf'), width=1280, height=540)

    def plot_spectra(self, fname_root, spectra='all', _range=None, ylim=None, title_text=None, backend='plotly',
                     plot_model=None, f=None, shade_reg=None):
        """
        Spectra.plot_spectra but incorporates the information from self.normalized.

        """
        print('Plotting spectra...')
        fmt = '.html' if backend == 'plotly' else '.pdf'
        if not os.path.exists(fname_root):
            os.makedirs(fname_root)
        if type(spectra) is str:
            if spectra == 'all':
                for i, item in enumerate(tqdm.tqdm(self)):
                    ttl = None if title_text is None else title_text[item]
                    if _range:
                        good = np.where((self[item].wave > _range[0]) & (self[item].wave < _range[1]))[0]
                        if good.size < 10:
                            continue
                    if f is not None:
                        fname = os.path.join(fname_root, self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    else:
                        fname = os.path.join(fname_root, self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    self[item].plot(fname=fname,
                                    backend=backend, _range=_range, ylim=ylim, title_text=ttl, plot_model=plot_model,
                                    shade_reg=shade_reg)
        else:
            for i, item in enumerate(tqdm.tqdm(self)):
                if item in spectra:
                    if item not in self or item not in title_text:
                        print(f'WARNING: {item} not found in stack!')
                        continue
                    if _range:
                        good = np.where((self[item].wave > _range[0]) & (self[item].wave < _range[1]))[0]
                        if good.size < 10:
                            continue
                    ttl = None if title_text is None else title_text[item]
                    if f is not None:
                        fname = os.path.join(fname_root,
                                             f'{f[i]:.3f}_' + self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    else:
                        fname = os.path.join(fname_root, self[item].name.replace(' ', '_') + '.spectrum' + fmt)
                    self[item].plot(fname=fname,
                                    backend=backend, _range=_range, ylim=ylim, title_text=ttl, plot_model=plot_model,
                                    shade_reg=shade_reg)
        print('Done.')

    def plot_hist(self, fname_base, plot_log=False, backend='plotly'):
        """
        Plot a histogram of the spectra in each bin.

        :param fname_base: str
            File name pattern.
        :param plot_log: boolean
            If True, makes the y-axis logarithmic.
        :param backend: str
            Whether to use matplotlib or plotly to plot stuff
        :return None:
        """
        fmt = '.html' if backend == 'plotly' else '.pdf'
        fname = fname_base + fmt
        widths = np.diff(self.bin_edges)
        midpts = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        nbins = len(widths)
        if backend == 'pyplot':
            fig, ax = plt.subplots()
            ax.bar(midpts, self.bin_counts, widths, align='center', color='rebeccapurple',
                   label='$n_{\\rm bins} = %d$' % nbins,
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
            xlabel = '$\\log(' + self.binned + ')$' if self.bin_log else self.binned
            fig.update_layout(
                xaxis_title=xlabel,
                yaxis_title='Number in bin',
                hovermode='x',
                xaxis=dict(
                    tickmode='array',
                    tickvals=self.bin_edges
                ),
                template='plotly_white'
            )

            fig.update_layout(
                font_family="Georgia, Times New Roman, Serif",
                # font_color="blue",
                title_font_family="Georgia, Times New Roman, Serif",
                # title_font_color="red",
                # legend_title_font_color="green"
            )
            fig.update_xaxes(title_font_family="Georgia, Times New Roman, Serif")
            fig.update_yaxes(title_font_family="Georgia, Times New Roman, Serif")

            if plot_log:
                fig.update_yaxes(
                    type="log",
                )
            fig.write_html(fname, include_mathjax="cdn")
            fig.write_image(fname.replace('.html', '.pdf'))

    def plot_agn(self, fname_base, bpt_x, bpt_y, bpt_xerr=None, bpt_yerr=None, labels=None, backend='plotly'):
        """
        Plot galaxies in the stack on a BPT-diagram, assuming galaxies have the appropriate BPT data to do so.

        :param fname_base: str
            Filename exlcuding the extension.
        :param bpt_x: str
            The name of the BPT ratio for the x-axis.
        :param bpt_y: str
            The name of the BPT ratio for the y-axis.
        :param bpt_xerr: optional, str
            The name of the error in the x-axis BPT ratio.  If not specified, no errorbars will be plotted.
        :param bpt_yerr: optional, str
            The name of the error in the y-axis BPT ratio.  If not specified, no errorbars will be plotted.
        :param labels: iterable
            List of strings specifying the names of the x and y axes.
        :param backend: str
            'plotly' or 'pyplot'
        :return:
        """
        fmt = '.html' if backend == 'plotly' else '.pdf'
        fname = fname_base + fmt
        data = np.array(
            [(self[i].data[bpt_x], self[i].data[bpt_y], self[i].data['agn_frac']) for i in range(len(self))])
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
            fig.add_trace(
                plotly.graph_objects.Scatter(x=k01_x, y=k01_y, line=dict(color='black', width=.5, dash='dash'),
                                             name='Kewley et al. 2001 Cutoff', showlegend=False))
            fig.update_layout(
                xaxis_title=xl,
                yaxis_title=yl,
                title='Reference point: ({:.5f},{:.5f})'.format(*self.agn_ref_pt),
                template='plotly_white'
            )

            fig.update_layout(
                font_family="Georgia, Times New Roman, Serif",
                # font_color="blue",
                title_font_family="Georgia, Times New Roman, Serif",
                # title_font_color="red",
                # legend_title_font_color="green"
            )
            fig.update_xaxes(title_font_family="Georgia, Times New Roman, Serif")
            fig.update_yaxes(title_font_family="Georgia, Times New Roman, Serif")

            fig.update_yaxes(
                range=(np.nanmin(y) - 0.05, np.nanmax(y) + 0.05),
                constrain='domain'
            )
            fig.update_xaxes(
                range=(np.nanmin(x) - 0.05, np.nanmax(x) + 0.05),
                constrain='domain'
            )
            fig.write_html(fname, include_mathjax="cdn")
            fig.write_image(fname.replace('.html', '.pdf'))

    def line_flux_report(self, fluxr_dict, line=6374, dw=10, norm_dw=(30, 30), plot_range=None, ratio_target=4,
                         plot_backend='plotly', path='',
                         agn_diagnostics=False, ylim=None, title_text_conf=None, title_text_snr=None, tag='',
                         conf_dict=None, conf_target=None, inspect=None, plot_spec='none'):
        """
        Plotting diagnostics for line flux ratio tests.

        :param fluxr_dict: dict
            Dictionary of results from a line flux test, as formatted by the calc_line_flux_ratios method.
        :param line: int, float
            The wavelength of the line that was integrated in fluxr_dict.
        :param dw: int, float
            Range of wavelengths to the left/right of line that was used in the integration of fluxr_dict.
        :param norm_dw: tuple
            Tuple with values for the number of angstroms to the left/right that the center of the comparison regions
            are from the center of the central region.  For plotting purposes.
        :param plot_range: tuple
            wavelength limits on the plotted data
        :param ratio_target: int, float
            A cutoff of the integrated line flux above which to consider a galaxies a 'detection'
        :param plot_backend: str
            'pyplot' or 'plotly'
        :param path: str
            Output path for plots.
        :param agn_diagnostics: boolean
            If True, calculate the Kewley et al. 2001 criteria for spectra and print comparisons to those detected by
            the line flux tests. Requires spectra to have BPT data ALREADY CALCULATED.
        :param ylim: optional, tuple
            Plot y-limits to be passed to the plot_spectra method.
        :param tag: optional, str
            A tag to append to the end of saved files.
        :param title_text_conf: optional, str
            Name of the a priori confidence value of a spectrum. Must be in the Spectrum's data dictionary.
        :param title_text_snr: optional, str
            Name of the SNR value of a spectrum. Must be in the Spectrum's data dictionary.
        :param conf_dict: optional, dict
            A priori confidence levels for each spectra, as formatted by the calc_line_flux_ratios method.
        :param conf_target: optional, float
            Required if conf_dict is provided and plot_spec is 'sorted'.  The target confidence level above which
            something is considered a true positive.
        :param inspect: str
            Key for a value in the data dictionary that has an additional property to be plotted on the color axis of
            the confidence covariance plots.
        :param plot_spec: str
            'none', 'all', 'detections', or 'sorted' to plot none, all, only those individual spectra that satisfy the
            detection criterium, or all in sorted sub-folders
        :return ss: list of np.ndarray
            The names of all the spectra that passed the detection criterium for each bin.
        """
        self._line_flux_diagnostics(fluxr_dict, line, dw, plot_backend=plot_backend, title_text_snr=title_text_snr,
                                    conf_dict=conf_dict,
                                    inspect=inspect, path=path)
        ss = []
        for i in range(len(fluxr_dict)):
            ratios = np.array([fluxr_dict[i][key] for key in fluxr_dict[i]])
            specnames = np.array([key for key in fluxr_dict[i]], dtype=object)

            good = [np.where((self[s].wave > line - dw) & (self[s].wave < line + dw))[0] for s in specnames]
            amps = np.zeros(specnames.size)
            amps_dict = {}
            for j, s in enumerate(specnames):
                if good[j].size > 1:
                    f = np.nanmax(self[s].flux[good[j]]) - 1
                    amps[j] = f
                    amps_dict[s] = f
                else:
                    amps_dict[s] = 0.

            if title_text_conf:
                confidences = np.array([self[sn].data[title_text_conf] for sn in specnames])
            # ind = ratios.argsort()[-5:][::-1]
            w = np.where(ratios >= ratio_target)[0]
            print(len(w), f' galaxies fulfill the threshold of {ratio_target}:')
            print(specnames[w])
            if agn_diagnostics:
                kew = np.array([self[i].data['agn_class'] for i in specnames])
                kw = np.where((kew is False) & (ratios >= ratio_target))[0]
                print(f'Of these, {len(kw)} did NOT satisfy the Kewley et al. 2001 criteria: ')
                print(specnames[kw])
            ss.append(specnames[w])
            if title_text_snr:
                snrs = np.array([self[si].data[title_text_snr] for si in specnames])
            tt = {s: r'$\mathcal{F}=%.3f$' % r for s, r in zip(specnames, ratios)} if not title_text_conf else \
                {s: r'$\mathcal{F}=%.3f$; $conf=%.3f$' % (r, c) for s, r, c in
                 zip(specnames, ratios, confidences)} if not title_text_snr else \
                    {s: r'$\mathcal{F}=%.3f$; $conf=%.3f$; $SNR=%.3f$, $A*SNR=%.3f$' % (r, c, snr, a * snr) for
                     s, r, c, snr, a in zip(specnames, ratios, confidences, snrs, amps)}
            if title_text_conf:
                tp = np.where((ratios >= ratio_target) & (confidences >= conf_target))[0]
                fp = np.where((ratios >= ratio_target) & (confidences < conf_target))[0]
                tn = np.where((ratios < ratio_target) & (confidences < conf_target))[0]
                fn = np.where((ratios < ratio_target) & (confidences >= conf_target))[0]
                ntp = len(tp)
                nfp = len(fp)
                ntn = len(tn)
                nfn = len(fn)
                tpsnr = np.nanmedian([self[ispec].data[title_text_snr] for ispec in specnames[tp]])
                fpsnr = np.nanmedian([self[ispec].data[title_text_snr] for ispec in specnames[fp]])
                tnsnr = np.nanmedian([self[ispec].data[title_text_snr] for ispec in specnames[tn]])
                fnsnr = np.nanmedian([self[ispec].data[title_text_snr] for ispec in specnames[fn]])
                tpasnr = np.nanmedian([self[ispec].data[title_text_snr] * amps_dict[ispec] for ispec in specnames[tp]])
                fpasnr = np.nanmedian([self[ispec].data[title_text_snr] * amps_dict[ispec] for ispec in specnames[fp]])
                tnasnr = np.nanmedian([self[ispec].data[title_text_snr] * amps_dict[ispec] for ispec in specnames[tn]])
                fnasnr = np.nanmedian([self[ispec].data[title_text_snr] * amps_dict[ispec] for ispec in specnames[fn]])
                with open(os.path.join(path, 'report_' + str(line) + '_' + tag + '.txt'), 'w') as file:
                    file.write('tp: ' + str(ntp) + ', med SNR: ' + str(tpsnr) + ', med A*SNR: ' + str(tpasnr) + '\n')
                    file.write('fp: ' + str(nfp) + ', med SNR: ' + str(fpsnr) + ', med A*SNR: ' + str(fpasnr) + '\n')
                    file.write('tn: ' + str(ntn) + ', med SNR: ' + str(tnsnr) + ', med A*SNR: ' + str(tnasnr) + '\n')
                    file.write('fn: ' + str(nfn) + ', med SNR: ' + str(fnsnr) + ', med A*SNR: ' + str(fnasnr) + '\n')
            else:
                with open(os.path.join(path, 'report_' + str(line) + '_' + tag + '.txt'), 'w') as file:
                    file.write('number integrated: ' + str(len(specnames)) + '\n')
                    file.write('detections: ' + str(len(specnames[w])) + '\n')
                    file.write('non-detections: ' + str(len(specnames) - len(specnames[w])) + '\n')
            if plot_spec != 'none':
                if plot_spec == 'sorted':
                    if conf_dict is not None:
                        tp = np.where((ratios >= ratio_target) & (confidences >= conf_target))[0]
                        fp = np.where((ratios >= ratio_target) & (confidences < conf_target))[0]
                        tn = np.where((ratios < ratio_target) & (confidences < conf_target))[0]
                        fn = np.where((ratios < ratio_target) & (confidences >= conf_target))[0]
                        name_list = [specnames[tp], specnames[fp], specnames[tn], specnames[fn]]
                        ratio_list = [ratios[tp], ratios[fp], ratios[tn], ratios[fn]]
                        path_list = ['true_positives_' + str(line), 'false_positives+' + str(line),
                                     'true_negatives_' + str(line), 'false_negatives_' + str(line)]
                    else:
                        nw = np.where(ratios < ratio_target)[0]
                        name_list = [specnames[w], specnames[nw]]
                        ratio_list = [ratios[w], ratios[nw]]
                        path_list = ['detections', 'non-detections']
                elif plot_spec == 'all':
                    name_list = [specnames]
                    ratio_list = [ratios]
                    path_list = ['spectra_' + str(line)]
                elif plot_spec == 'detections':
                    name_list = [specnames[w]]
                    ratio_list = [ratios[w]]
                    path_list = ['detections_' + str(line)]
                else:
                    raise ValueError('invalid plot_spec option')
                for names, rts, pathi in zip(name_list, ratio_list, path_list):
                    shade = [(line - dw, line + dw), (line - norm_dw[0] - dw, line - norm_dw[0] + dw),
                             (line + norm_dw[1] - dw, line + norm_dw[1] + dw)]
                    self.plot_spectra(os.path.join(path, pathi), names, _range=plot_range, ylim=ylim,
                                      backend=plot_backend,
                                      title_text=tt, f=None, shade_reg=shade)
        return ss

    def _line_flux_diagnostics(self, fluxr_dict, line=6374, dw=5, title_text_snr=None, plot_backend='pyplot',
                               conf_dict=None,
                               inspect=None, path=''):
        # 10^-17 erg s^-1 cm^-2 (not per angstrom after integration)
        # Get data
        # lengths = [len(fluxr_dict[fi]) for fi in fluxr_dict]
        ratios = []
        specnames = []
        # stack_fluxes = []
        amps = []
        snrs = []
        inspections = []
        for i in range(len(fluxr_dict)):
            ratios.append(np.array([fluxr_dict[i][fi] for fi in fluxr_dict[i]]))
            specnames.append(np.array([fi for fi in fluxr_dict[i]]))

            good = [np.where((self[s].wave > line - dw) & (self[s].wave < line + dw))[0] for s in specnames[i]]
            ampi_temp = np.zeros(specnames[i].size)
            for j, s in enumerate(specnames[i]):
                if good[j].size > 1:
                    ampi_temp[j] = np.nanmax(self[s].flux[good[j]]) - 1
            amps.append(ampi_temp)

            if inspect:
                inspections.append(np.array([self[si].data[inspect] for si in specnames[i]]))
            if title_text_snr:
                snrs.append(np.array([self[si].data[title_text_snr] for si in specnames[i]]))
            # stack_fluxes.append(fluxr_dict[i]['stack'][0])

        # Get mins/maxes
        # lstr = len(list(fluxr_dict[0].keys())[0])
        mins = np.full((2, len(fluxr_dict)), fill_value='', dtype=object)
        maxs = np.full_like(mins, fill_value='', dtype=object)
        minr = np.full((2, len(fluxr_dict)), fill_value=np.nan, dtype=float)
        maxr = np.full_like(minr, fill_value=np.nan, dtype=float)
        for j in range(len(fluxr_dict)):
            minrind = np.nanargmin(ratios[j])
            min2rind = np.nanargmin(np.delete(ratios[j], minrind))
            maxrind = np.nanargmax(ratios[j])
            max2rind = np.nanargmax(np.delete(ratios[j], maxrind))
            maxr[:, j] = (ratios[j][maxrind], np.delete(ratios[j], maxrind)[max2rind])
            minr[:, j] = (ratios[j][minrind], np.delete(ratios[j], minrind)[min2rind])
            maxs[:, j] = (specnames[j][maxrind], np.delete(specnames[j], maxrind)[max2rind])
            mins[:, j] = (specnames[j][minrind], np.delete(specnames[j], minrind)[min2rind])

        reg = (line - dw * 2, line + dw * 2)
        if plot_backend == 'pyplot':
            # Set up a gridspec
            for k in range(len(fluxr_dict)):
                fig = plt.figure(constrained_layout=True)
                gs = fig.add_gridspec(3, 3)
                # Main flux ratio plot
                ratioplot = fig.add_subplot(gs[0:2, 0:2])
                ratioplot.hist(ratios[k], bins=np.arange(0, 3.2, 0.2), density=False)
                ratioplot.set_yscale('log')
                ratioplot.set_xlabel('Line flux / Stacked line flux')
                ratioplot.set_ylabel('Number in bin')
                # ratioplot.set_title('Stacked line flux $= %.3f' % stack_fluxes[k])
                # ratioplot.set_ylim(0, 3)

                small = []
                big = []
                for m in range(2):
                    # 2 smallest line ratios' profiles
                    good1 = np.where((reg[0] < self[mins[m, k]].wave) & (self[mins[m, k]].wave < reg[1]))[0]
                    small1 = fig.add_subplot(gs[2, m])
                    small1.plot(self[mins[m, k]].wave[good1], self[mins[m, k]].flux[good1], 'k-')
                    small1.fill_between(self[mins[m, k]].wave[good1],
                                        self[mins[m, k]].flux[good1] - self[mins[m, k]].error[good1],
                                        self[mins[m, k]].flux[good1] + self[mins[m, k]].error[good1],
                                        color='mediumaquamarine', alpha=0.5)
                    small1.axvline(line, linestyle='--', color='k')
                    small1.axvspan(line - dw, line + dw, color='slategrey', alpha=0.5)
                    ttl = mins[m, k] if len(mins[m, k]) <= 10 else mins[m, k][0:10] + '...'
                    small1.set_title(ttl)
                    small1.set_xticks([line - dw, line + dw])
                    if m == 0:
                        small1.set_ylabel('Norm. Flux')
                    elif m == 1:
                        small1.set_xlabel('Wavelength [${\\rm \\AA}$]')
                    small.append(small1)
                    # 2 largest line ratios' profiles
                    good2 = np.where((reg[0] < self[maxs[m, k]].wave) & (self[maxs[m, k]].wave < reg[1]))[0]
                    big1 = fig.add_subplot(gs[m, 2])
                    big1.plot(self[maxs[m, k]].wave[good2], self[maxs[m, k]].flux[good2], 'k-')
                    big1.fill_between(self[maxs[m, k]].wave[good2],
                                      self[maxs[m, k]].flux[good2] - self[maxs[m, k]].error[good2],
                                      self[maxs[m, k]].flux[good2] + self[maxs[m, k]].error[good2],
                                      color='mediumaquamarine', alpha=0.5)
                    big1.axvline(line, linestyle='--', color='k')
                    big1.axvspan(line - dw, line + dw, color='slategrey', alpha=0.5)
                    ttl = maxs[m, k] if len(maxs[m, k]) <= 10 else maxs[m, k][0:10] + '...'
                    big1.set_title(ttl)
                    big1.set_xticks([line - dw, line + dw])
                    big.append(big1)

                # Stack line profile
                # stack = fig.add_subplot(gs[2, 2])
                # good = np.where((reg[0] < self.universal_grid[k]) & (self.universal_grid[k] < reg[1]))[0]
                # stack.plot(self.universal_grid[k][good], self.stacked_flux[k][good], 'k-')
                # stack.fill_between(self.universal_grid[k][good],
                #                    self.stacked_flux[k][good]-self.stacked_err[k][good],
                #                    self.stacked_flux[k][good]+self.stacked_err[k][good],
                #                    color='mediumaquamarine', alpha=0.5)
                # stack.axvline(line, linestyle='--', color='k')
                # stack.axvspan(line-dw, line+dw, color='slategrey', alpha=0.5)
                # stack.set_xticks([line-dw, line+dw])

                # Axis sharing
                big[0].sharex(big[1])
                # stack.sharex(big[0])
                small[0].sharex(big[0])
                small[1].sharex(small[0])
                big[0].sharey(big[1])
                # stack.sharey(big[0])
                small[0].sharey(big[0])
                small[1].sharey(small[0])

                # stack.set_title('Stack')
                fig.savefig(path + os.sep + 'line_flux_ratios_' + str(k) + '.pdf', dpi=300)
                plt.close()

            if conf_dict:
                confidences = []

                for k in range(len(fluxr_dict)):
                    confidences.append(np.array([conf_dict[k][fi] for fi in conf_dict[k] if fi != 'stack']))

                    # Sort
                    isort = np.argsort(confidences[k])
                    confidences[k] = confidences[k][isort]
                    ratios[k] = ratios[k][isort]
                    amps[k] = amps[k][isort]
                    snrs[k] = snrs[k][isort]
                    asnr = amps[k] * snrs[k]
                    if inspect:
                        inspections[k] = inspections[k][isort]

                    std = np.nanstd(ratios[k])
                    median = np.nanmedian(ratios[k])
                    good = np.where((np.abs(ratios[k] - median) < 3 * std) & (
                            np.isfinite(confidences[k]) & np.isfinite(ratios[k])))[0]
                    ratios[k] = ratios[k][good]
                    confidences[k] = confidences[k][good]
                    asnr = asnr[good]

                    # ntop = int(.2*len(ratios[k]))
                    # rsort = np.argsort(ratios[k])
                    # top20 = ratios[k][rsort][len(ratios[k])-ntop:]
                    # top20c = confidences[k][rsort][len(ratios[k])-ntop:]

                    A = np.vstack((confidences[k], np.ones_like(confidences[k]))).T
                    m, c = np.linalg.lstsq(A, ratios[k], rcond=None)[0]
                    x_model = np.linspace(confidences[k][0], confidences[k][-1], 1000)
                    y_model = m * x_model + c

                    fig, ax = plt.subplots()
                    dataplot = ax.plot(confidences[k], ratios[k], '.', color=asnr if not inspect else inspections[k],
                                       cmap='winter')
                    fig.colorbar(dataplot, ax=ax, label='$A*SNR$' if not inspect else 'Inspections')
                    # ax.plot(top20c, top20, '.', color='cyan', label='Top 20%')
                    ax.plot(x_model, y_model, 'k--')
                    ax.legend()
                    ax.set_title(r'Linear Least Squares Fit: $m=%.3f$, $b=%.3f$' % (m, c))
                    ax.set_xlabel('Confidence Level')
                    ax.set_ylabel(r'Line Flux Ratio Parameter $\mathcal{F}$')
                    # ax.set_ylim(-10, 10)
                    fig.savefig(path + os.sep + 'line_flux_confidence_covar_' + str(k) + '.pdf', dpi=300)
                    plt.close()

        elif plot_backend == 'plotly':
            for k in range(len(fluxr_dict)):
                ttl1 = maxs[0, k] if len(maxs[0, k]) <= 10 else maxs[0, k][0:10] + '...'
                ttl2 = maxs[1, k] if len(maxs[1, k]) <= 10 else maxs[1, k][0:10] + '...'
                ttl3 = mins[0, k] if len(mins[0, k]) <= 10 else mins[0, k][0:10] + '...'
                ttl4 = mins[1, k] if len(mins[1, k]) <= 10 else mins[1, k][0:10] + '...'
                fig = plotly.subplots.make_subplots(rows=3, cols=3,
                                                    specs=[[{"rowspan": 2, "colspan": 2}, None, {}],
                                                           [None, None, {}],
                                                           [{}, {}, {}]],
                                                    subplot_titles=['',
                                                                    ttl1, ttl2, ttl3, ttl4, 'Stack'])
                fig.add_trace(plotly.graph_objects.Histogram(x=ratios[k], xbins=dict(start=0, end=max(ratios[k]),
                                                                                     size=max(ratios[k])/10)), row=1,
                              col=1)
                fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                  marker_line_width=0.0, opacity=0.8, row=1, col=1)
                fig.update_layout(
                    yaxis=dict(type='log'),
                    yaxis_title_text='Number in bin',
                    xaxis_title_text='Line flux / stacked line flux',
                    template='plotly_white'
                    # title_text='Stacked line flux $= %.3f \\times 10^{-17}$ erg s$^{-1}$ cm$^{-2}$' % stack_fluxes[k],
                )
                linewidth = .5
                maxy = -9999
                miny = 9999
                for m in range(2):
                    good1 = np.where((reg[0] < self[mins[m, k]].wave) & (self[mins[m, k]].wave < reg[1]))[0]
                    fig.add_trace(
                        plotly.graph_objects.Scatter(x=self[mins[m, k]].wave[good1], y=self[mins[m, k]].flux[good1],
                                                     line=dict(color='black', width=linewidth),
                                                     name='Data', showlegend=False), row=3, col=m + 1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=self[mins[m, k]].wave[good1],
                                                               y=self[mins[m, k]].flux[good1] + self[mins[m, k]].error[
                                                                   good1],
                                                               line=dict(color='#60dbbd', width=0),
                                                               fillcolor='rgba(96, 219, 189, 0.6)',
                                                               name='Upper Bound', showlegend=False), row=3, col=m + 1)
                    fig.add_trace(plotly.graph_objects.Scatter(x=self[mins[m, k]].wave[good1],
                                                               y=self[mins[m, k]].flux[good1] - self[mins[m, k]].error[
                                                                   good1],
                                                               line=dict(color='#60dbbd', width=0),
                                                               fillcolor='rgba(96, 219, 189, 0.6)',
                                                               fill='tonexty', name='Lower Bound', showlegend=False),
                                  row=3, col=m + 1)
                    fig.add_vline(x=line, line_width=2 * linewidth, line_dash='dot', line_color='#226666', row=3,
                                  col=m + 1)
                    n = m + 4
                    fig['layout']['xaxis' + str(n)]['tickmode'] = 'array'
                    fig['layout']['xaxis' + str(n)]['tickvals'] = [line - dw, line + dw]
                    # fig['layout']['title'+str(n)]['text'] = mins[m, k]
                    if m == 0:
                        fig['layout']['yaxis' + str(n)]['title_text'] = 'Norm. Flux'
                    elif m == 1:
                        fig['layout']['xaxis' + str(n)]['title_text'] = 'Wavelength [&#8491;]'
                    maxiy = np.nanmax(self[mins[m, k]].flux[good1]) + .5
                    miniy = np.nanmin(self[mins[m, k]].flux[good1]) - .5
                    if maxiy > maxy:
                        maxy = maxiy
                    if miniy < miny:
                        miny = miniy
                    good2 = np.where((reg[0] < self[maxs[m, k]].wave) & (self[maxs[m, k]].wave < reg[1]))[0]
                    fig.add_trace(
                        plotly.graph_objects.Scatter(x=self[maxs[m, k]].wave[good2], y=self[maxs[m, k]].flux[good2],
                                                     line=dict(color='black', width=linewidth),
                                                     name='Data', showlegend=False), row=m + 1, col=3)
                    fig.add_trace(plotly.graph_objects.Scatter(x=self[maxs[m, k]].wave[good2],
                                                               y=self[maxs[m, k]].flux[good2] + self[maxs[m, k]].error[
                                                                   good2],
                                                               line=dict(color='#60dbbd', width=0),
                                                               fillcolor='rgba(96, 219, 189, 0.6)',
                                                               name='Upper Bound', showlegend=False), row=m + 1, col=3)
                    fig.add_trace(plotly.graph_objects.Scatter(x=self[maxs[m, k]].wave[good2],
                                                               y=self[maxs[m, k]].flux[good2] - self[maxs[m, k]].error[
                                                                   good2],
                                                               line=dict(color='#60dbbd', width=0),
                                                               fillcolor='rgba(96, 219, 189, 0.6)',
                                                               fill='tonexty', name='Lower Bound', showlegend=False),
                                  row=m + 1, col=3)
                    fig.add_vline(x=line, line_width=2 * linewidth, line_dash='dot', line_color='#226666', row=m + 1,
                                  col=3)
                    n2 = m + 2
                    fig['layout']['xaxis' + str(n2)]['tickmode'] = 'array'
                    fig['layout']['xaxis' + str(n2)]['tickvals'] = [line - dw, line + dw]
                    # fig['layout']['title'+str(n2)]['text'] = maxs[m, k]
                    maxiy = np.nanmax(self[maxs[m, k]].flux[good2]) + .5
                    miniy = np.nanmin(self[maxs[m, k]].flux[good2]) - .5
                    if maxiy > maxy:
                        maxy = maxiy
                    if miniy < miny:
                        miny = miniy

                # good = np.where((reg[0] < self.universal_grid[k]) & (self.universal_grid[k] < reg[1]))[0]
                # fig.add_trace(plotly.graph_objects.Scatter(x=self.universal_grid[k][good], y=self.stacked_flux[k][good],
                #                                            line=dict(color='black', width=linewidth),
                #                                            name='Data', showlegend=False), row=3, col=3)
                # fig.add_trace(plotly.graph_objects.Scatter(x=self.universal_grid[k][good],
                #                                            y=self.stacked_flux[k][good] + self.stacked_err[k][good],
                #                                            line=dict(color='#60dbbd', width=0),
                #                                            fillcolor='rgba(96, 219, 189, 0.6)',
                #                                            name='Upper Bound', showlegend=False), row=3, col=3)
                # fig.add_trace(plotly.graph_objects.Scatter(x=self.universal_grid[k][good],
                #                                            y=self.stacked_flux[k][good] - self.stacked_err[k][good],
                #                                            line=dict(color='#60dbbd', width=0),
                #                                            fillcolor='rgba(96, 219, 189, 0.6)',
                #                                            fill='tonexty', name='Lower Bound', showlegend=False), row=3, col=3)
                # fig.add_vline(x=line, line_width=2 * linewidth, line_dash='dot', line_color='#226666', row=3, col=3)
                # fig['layout']['text'+str(6)] = 'Stack'
                fig['layout']['xaxis' + str(6)]['tickmode'] = 'array'
                fig['layout']['xaxis' + str(6)]['tickvals'] = [line - dw, line + dw]
                for i in range(2, 6):
                    fig['layout']['yaxis' + str(i)]['range'] = (miny, maxy)
                    fig['layout']['yaxis' + str(i)]['constrain'] = 'domain'
                    fig['layout']['xaxis' + str(i)]['range'] = (line - 2 * dw, line + 2 * dw)
                    fig['layout']['xaxis' + str(i)]['constrain'] = 'domain'
                    fig.add_shape(type='rect', xref='x' + str(i), yref='y' + str(i), x0=line - dw, y0=miny,
                                  x1=line + dw, y1=maxy, fillcolor='lightgrey', opacity=0.5,
                                  line_width=0, layer='below')

                fig.update_layout(
                    font_family="Georgia, Times New Roman, Serif",
                    # font_color="blue",
                    title_font_family="Georgia, Times New Roman, Serif",
                    # title_font_color="red",
                    # legend_title_font_color="green"
                )
                fig.update_xaxes(title_font_family="Georgia, Times New Roman, Serif")
                fig.update_yaxes(title_font_family="Georgia, Times New Roman, Serif")

                fig.write_html(path + os.sep + 'line_flux_ratios_' + str(k) + '.html', include_mathjax="cdn")
                fig.write_image(path + os.sep + 'line_flux_ratios_' + str(k) + '.pdf')

            if conf_dict:
                confidences = []

                for k in range(len(fluxr_dict)):
                    confidences.append(np.array([conf_dict[k][fi] for fi in conf_dict[k] if fi != 'stack']))

                    # Sort
                    isort = np.argsort(confidences[k])
                    confidences[k] = confidences[k][isort]
                    ratios[k] = ratios[k][isort]
                    amps[k] = amps[k][isort]
                    snrs[k] = snrs[k][isort]
                    asnr = amps[k] * snrs[k]
                    if inspect:
                        inspections[k] = inspections[k][isort]

                    std = np.nanstd(ratios[k])
                    median = np.nanmedian(ratios[k])
                    good = np.where((np.abs(ratios[k] - median) < 3 * std) & (
                            np.isfinite(confidences[k]) & np.isfinite(ratios[k])))[0]
                    ratios[k] = ratios[k][good]
                    confidences[k] = confidences[k][good]
                    asnr = asnr[good]

                    # ntop = int(.2 * len(ratios[k]))
                    # rsort = np.argsort(ratios[k])
                    # top20 = ratios[k][rsort][len(ratios[k]) - ntop:]
                    # top20c = confidences[k][rsort][len(ratios[k]) - ntop:]

                    A = np.vstack((confidences[k], np.ones_like(confidences[k]))).T
                    m, c = np.linalg.lstsq(A, ratios[k], rcond=None)[0]
                    x_model = np.linspace(confidences[k][0], confidences[k][-1], 1000)
                    y_model = m * x_model + c

                    fig = plotly.subplots.make_subplots(rows=1, cols=1)
                    fig.add_trace(plotly.graph_objects.Scatter(
                        x=confidences[k], y=ratios[k], mode='markers',
                        marker=dict(size=4,
                                    color=asnr if not inspect else inspections[k],
                                    colorbar=dict(title="$A \\times SNR$" if not inspect else "Inspections"),
                                    colorscale="ice" if not inspect else "spectral",
                                    cmin=0 if not inspect else -1,
                                    cmax=4 if not inspect else 1),
                        showlegend=False))
                    # fig.add_trace(plotly.graph_objects.Scatter(x=top20c, y=top20, mode='markers',
                    #     marker=dict(size=4, color='#48CADB'), showlegend=False))
                    fig.add_trace(plotly.graph_objects.Scatter(x=x_model, y=y_model,
                                                               line=dict(color='black', width=.5, dash='dash'),
                                                               showlegend=False))
                    fig.update_layout(
                        template='plotly_white',
                        title='Linear Least Squares Fit: m=%.3f, b=%.3f' % (m, c),
                        xaxis_title='Confidence Level',
                        yaxis_title='${\\rm Line Flux Ratio Parameter}\\ \\mathcal{F}$',
                        # yaxis_range=(-10, 10),
                        # yaxis_constrain='domain'
                    )

                    fig.update_layout(
                        font_family="Georgia, Times New Roman, Serif",
                        # font_color="blue",
                        title_font_family="Georgia, Times New Roman, Serif",
                        # title_font_color="red",
                        # legend_title_font_color="green"
                    )
                    fig.update_xaxes(title_font_family="Georgia, Times New Roman, Serif")
                    fig.update_yaxes(title_font_family="Georgia, Times New Roman, Serif")

                    fig.write_html(path + os.sep + 'line_flux_confidence_covar_' + str(k) + '.html',
                                   include_mathjax="cdn")
                    fig.write_image(path + os.sep + 'line_flux_confidence_covar_' + str(k) + '.pdf')

        else:
            raise NotImplementedError

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

    def save_fits(self, filepath, bin_num=(0,)):
        """
        Save the stacked spectrum for a specific bin to a FITS file.

        :param filepath: str
            The filepath to save to.
        :param bin_num: tuple
            Which bins to save.  Default is (0,).
        :return:
        """
        hdu = astropy.io.fits.HDUList()
        for b in bin_num:
            header = astropy.io.fits.Header()
            # Spectrum has already been corrected for redshift and extinction, so make these 0
            header['z'] = 0.
            header['ebv'] = 0.
            # Rem: "Since we aren't really concerned with the exact values of these quantities
            # (since we're dealing with stacks, their actual kinematic quantities are meaningless),
            # you can set the FWHM resolution to something small like 0.1 A. "
            header['fwhm'] = 0.1
            hdu.append(astropy.io.fits.PrimaryHDU(data=self.stacked_flux[b], header=header))
            hdu.append(astropy.io.fits.PrimaryHDU(data=self.universal_grid[b]))
            hdu.append(astropy.io.fits.PrimaryHDU(data=self.stacked_err[b]))
        hdu.writeto(filepath)
        return hdu

    def __repr__(self):
        s = f"A collection of {len(self)} stacked spectra.\n"
        s += f"Corrected:  \t {self.corrected}\n"
        s += f"Stacked:    \t {True if len(self.stacked_flux) > 0 else False}\n"
        s += f"Binned:     \t {'log_' if self.bin_log else ''}{self.binned}"
        return s
