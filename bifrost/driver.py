from bifrost import spectrum, utils
import os
import tqdm
from joblib import Parallel, delayed


def driver(data_path, out_path=None, n_jobs=-1, save_pickle=True, save_json=False, plot_backend='plotly',
         plot_spec=None, limits=None):
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
    :return stack: Stack
        The Stack object.
    """
    if not out_path:
        out_path = 'data.stacked.' + utils.gen_datestr(True)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path += os.sep

    all_spectra = utils.get_filepaths_from_parent(data_path, 'fits')
    if limits:
        all_spectra = all_spectra[limits[0]:limits[1]]
    stack = spectrum.Stack()

    def make_spec(filepath):
        ispec = spectrum.Spectrum.from_fits(filepath, name=filepath.split(os.sep)[-1])
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