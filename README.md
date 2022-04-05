# BIFR&#xd6;ST
## Black hole Investigations of Forbidden-line Radiation in Optical SpecTra

This code aims to connect galaxy spectra to coronal line detections, just as 
the burning rainbow bridge it is named after connects Asgard and Midgard, 
by providing a means of searching for them via a framework for analyzing spectra
with simple tests and a stacking procedure.

## Installation
After cloning the repository, you can install locally via pip: ` pip install . `
<br>Or alternatively, ` python setup.py install ` All of the dependencies are listed in the
`requirements.txt` file, and should be installed automatically.  These dependencies are:

- `numpy < 1.21, >= 1.17`
- `numba >= 0.53.1`
- `pandas >= 1.3.2`
- `tqdm >= 4.61.2`
- `matplotlib >= 3.4.2`
- `plotly >= 4.14.3`
- `astropy >= 4.2.1`
- `astroquery >= 0.4.2`
- `spectres >= 2.1.1`
- `scipy >= 1.7.0`
- `joblib >= 1.0.1`
- `PyAstronomy >= 0.16.0`
- `kaleido >= 0.2.1`

<br> I can't guarantee this package will work  for any versions of these requirements
outside the suggestions above.

## Documentation Summary
Most of the functions of this module are built around three core objects: `Spectrum`, `Spectra`, 
and `Stack`. 
Please see the source code for more detailed docstrings for each class and method.

### Spectrum
The Spectrum class stores a single object's spectrum, including an array of wavelengths, fluxes, and 1-sigma errors.
Information about redshift, coordinates, and extinction can also be stored and (if available) used to apply corrections to
the spectrum.  The basic constructor is:
<br>`class Spectrum(wave, flux, error, redshift=None, velocity=None, ra=None, dec=None, ebv=None, name='Generic',
                 output_path=None, **data)`

<br>There are also constructors available for creating an object directly from an SDSS-formatted FITS file:
`Spectrum.from_fits`, and from completely simulated data: `Spectrum.simulated`.

As a side note, basic arithmetic operators (`+, -, *, /`) may be performed between Spectrum objects.  Doing so will create a new 
Spectrum object with the flux combined as expected, and the error will be propagated as well.  This is only possible if the two
objects being added have identical wavelength arrays and are both corrected, and they must also either both or neither be normalized.

### Spectra
The Spectra object is a child of both Spectrum and dict.  It is a convenient way of storing many Spectrum objects together
in a dictionary, with methods for correcting and plotting each Spectrum in the dictionary.  Also as a convenience, the object
can be indexed like a list or a dictionary.  i.e. if the first element in the dictionary has a key 'Spectrum A', it can be
obtained either using the key 'Spectrum A', or the index 0.  There is also a method called `to_numpy` for converting any attributes
into numpy arrays.  Its basic constructor is no different from a typical dictionary:
<br>`class Spectra()`
<br>and items can be added with the `add_spec` method:
`Spectra.add_spec(SpectrumA)`

### Stack
The main attraction, the Stack object, is a child of the Spectra object, thus making it also dictionary-like. This class
has methods for stacking all the spectra inside it together, binning the spectra and then stacking each bin, calculating line flux
ratios, and many plotting methods.

To perform the stacking procedure, (which corrects all spectra in the Stack for redshift and extinction, normalizes them, 
resamples their fluxes over a uniform wavelength grid, and then coadds them using a weighted mean, with inverse variances used
as weights), one must call the Stack object instance directly, i.e. if we have two Spectrum objects A and B:
```python
# Create the stack object
stack = bifrost.Stack()
# Create spectrum objects
A = bifrost.Spectrum(wave1, flux1, error1, redshift1)
B = bifrost.Spectrum(wave2, flux2, error2, redshift2)
# Append Spectrum objects to the stack
stack.add_spec(A)
stack.add_spec(B)
# Call the stacking procedure with the default arguments
stack()
```

## Examples
See the example files in the examples folder to get a sense of how to use the code.
Feel free to also run the unit tests located in `unit_test.py`.