import os
import pickle
import json

import numpy as np
import tqdm
from joblib import Parallel, delayed

from bifrost import spectrum, utils, filters


def driver(data_path, out_path=None, n_jobs=-1, save_pickle=True, save_json=False, plot_backend='plotly',
         plot_spec=None, limits=None, _filters=None, name_by='folder'):
    """
    The main driver for the stacking code.

    :param data_path: str
        The path to the parent folder containing fits files, or subfolders with fits files.
    :param out_path: str
        The output path to save output plots and pickles/jsons to.  Default is "data.stacked.YYYYMMDD_HHMMSS"
    :param n_jobs: int
        The number of jobs to run in parallel when reading in fits files.  Default is -1, meaning
        as many jobs as are allowed to run in parallel.
    :param save_pickle: bool
        Whether or not to save the Stack object as a pickle file.  Default is true.
    :param save_json: bool
        Whether or not to save the Stack object as a json file.  Default is false.
    :param plot_backend: str
        May be 'pyplot' to use pyplot or 'plotly' to use plotly for plotting.  Default is 'plotly'.
    :param plot_spec: str, iterable
        Which spectra to plot individually.  Default is None, which doesn't plot any.
    :param limits: tuple
        Limit to only use data in the range of these indices.
    :param _filters: str, iterable
        Filter objects to be applied to the Stack.
    :param name_by: str
        "folder" or "file" : how to specify object keys, based on the name of the fits file or the folder that the fits
        file is in.
    :return stack: Stack
        The Stack object.
    """
    # Create output paths
    if not out_path:
        out_path = 'data.stacked.' + utils.gen_datestr(True)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path += os.sep

    # Gather spectra paths
    all_spectra = utils.get_filepaths_from_parent(data_path, 'fits')
    if limits:
        all_spectra = all_spectra[limits[0]:limits[1]]

    # Configure filter objects
    filter_list = []
    if not _filters:
        _filters = []
    if type(_filters) is str:
        _filters = [_filters]
    for _filter in _filters:
        filter_list.append(filters.Filter.from_str(_filter))
    stack = spectrum.Stack(filters=filter_list)

    assert name_by in ('file', 'folder'), "name_by must be one of ['file', 'folder']"
    def make_spec(filepath):
        ind = -1 if name_by == 'file' else -2
        ispec = spectrum.Spectrum.from_fits(filepath, name=filepath.split(os.sep)[ind])
        return ispec

    print('Loading in spectra...')
    specs = Parallel(n_jobs=n_jobs)(delayed(make_spec)(fpath) for fpath in tqdm.tqdm(all_spectra))
    for ispec in specs:
        stack.add_spec(ispec)

    stack()
    if plot_spec:
        stack.plot_spectra(out_path, spectra=plot_spec, backend=plot_backend)
    format = '.html' if plot_backend == 'plotly' else '.pdf'
    stack.plot_stacked(out_path+'stacked_plot'+format, backend=plot_backend)
    if save_pickle:
        stack.save_pickle(out_path+'stacked_data.pkl')
    if save_json:
        stack.save_json(out_path+'stacked_data.json')

    return stack


def plotter(stack_path, out_path=None, plot_backend='plotly', plot_spec=None):
    """
    Replot the stacked spectra.

    :param stack_path: str
        Path to a pickle or json file containing the stack object.
    :param out_path: str
        Folder / file name to save the new files to.
    :param plot_backend: str
        May be 'pyplot' to use pyplot or 'plotly' to use plotly for plotting.  Default is 'plotly'.
    :param plot_spec: str, iterable
        Spectra to plot individually
    :return None:
    """
    format = stack_path.split('.')[-1]
    plot_format = '.html' if plot_backend == 'plotly' else '.pdf'
    if not out_path:
        out_path = os.path.dirname(stack_path)
    if format == 'json':
        def json_stack_hook(_dict):
            options = dict(
                r_v=_dict['r_v'],
                gridspace=_dict['gridspace'],
                tolerance=_dict['tolerance'],
                norm_region=_dict['norm_region'],
                default_filters=_dict['default_filters']
            )
            fs = [filters.Filter.from_str(ff) for ff in _dict['filters']]
            return spectrum.Stack(universal_grid=np.array(_dict['universal_grid']), stacked_flux=np.array(_dict['stacked_flux']),
                                  stacked_err=np.array(_dict['stacked_err']), resampled=_dict['resampled'], normalized=_dict['normalized'],
                                  filters=fs, **options)

        stack = json.load(open(stack_path, 'r'), object_hook=json_stack_hook)
    elif format == 'pkl':
        stack = pickle.load(open(stack_path, 'rb'))
    else:
        raise ValueError(f"Cannot read object file type: {format}")

    if not plot_spec:
        stack.plot_stacked(os.path.join(out_path, 'stacked_plot'+plot_format), backend=plot_backend)
    else:
        if format == 'json':
            raise AttributeError("Cannot read individual spectra from a saved json file, only the stacked data "
                                 "is saved.")
        stack.plot_spectra(out_path, spectra=plot_spec, backend=plot_backend)


def edit_config(**options):
    """
    Edit the default configuration options for the Stack object.
    :param options: dict
        Keyword arguments corresponding to each option
    :return None:
    """
    # Load in the current config file
    config_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.json')
    blueprints = json.load(open(config_path, 'r'))

    # Edit the options based on arguments
    for option in options:
        value = options[option]
        if option == 'r_v':
            assert value >= 0, "r_v must be positive!"
        elif option == 'gridspace':
            assert value > 0, "gridspace must be nonzero!"
        elif option == 'tolerance':
            assert value > 0, "tolerance must be nonzero!"
        elif option == 'norm_region' and value is not None:
            assert value[0] < value[1], "right bound must be larger than left bound!"
        blueprints[option] = value

    # Rewrite the config file with the new options
    serialized = json.dumps(blueprints, indent=4)
    with open(config_path, 'w') as handle:
        handle.write(serialized)
