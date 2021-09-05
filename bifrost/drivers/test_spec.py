from bifrost import spectrum, utils
import os
import tqdm
from joblib import Parallel, delayed

subdir = 'data.stacked.26000' + os.sep

bifrost_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')) + os.sep
all_spectra = utils.get_filepaths_from_parent(bifrost_path+'data/', 'fits')
# all_spectra = all_spectra[0:1000]
stack = spectrum.Stack()


def make_spec(filepath):
    ispec = spectrum.Spectrum.from_fits(filepath, name=filepath.split(os.sep)[-2])
    return ispec


if __name__ == '__main__':
    print('Loading in spectra...')
    specs = Parallel(n_jobs=-1)(delayed(make_spec)(fpath) for fpath in tqdm.tqdm(all_spectra))
    for ispec in specs:
        stack.add_spec(ispec)

    stack()
    stack.plot_spectra(bifrost_path+subdir, spectra=[0, 1, 2, 3, 4, -1, -2, -3, -4, -5])
    stack.plot_stacked(bifrost_path+subdir+'stacked_plot.html')
    stack.save_pickle(bifrost_path+subdir+'stacked_data.pkl')
    stack.save_json(bifrost_path+subdir+'stacked_data.json')
