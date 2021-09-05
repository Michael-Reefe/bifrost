from bifrost import spectrum, utils
import glob
import os
import time
import tqdm

bifrost_path = '/Users/mreefe/Library/Mobile Documents/com~apple~CloudDocs/Astrophysics/bifrost/'
all_spectra = utils.get_filepaths_from_parent(bifrost_path+'data/', 'fits')
all_spectra = all_spectra[0:100]

print('Loading in spectra...this may take a while.')
stack = spectrum.Stack()
for path in tqdm.tqdm(all_spectra):
    ispec = spectrum.Spectrum.from_fits(path, name=path.split(os.sep)[-2])
    stack.add_spec(ispec)

stack()
stack.plot_spectra(bifrost_path+'data.stacked.100/', spectra=[0, 1, 2, 3, 4])
stack.plot_stacked(bifrost_path+'data.stacked.100/stacked_plot.html')
stack.save_pickle(bifrost_path+'data.stacked.100/stacked_data.pkl')
stack.save_json(bifrost_path+'data.stacked.100/stacked_data.json')
