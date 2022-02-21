# First import some base modules
import os
import pickle
import sys
# Import numpy
import numpy as np
# Import the bifrost module
# NOTE: this WILL NOT WORK until you have installed the module with `pip install .` from the terminal.
import bifrost as bf

# Set up some configurations and file paths
# First, get your data from whatever method you please, in this case we just use simulated data
wave = np.arange(4000, 7000, 1)
z = 0.001
flux = np.random.normal(1, 0.1, (wave.size, 10))
error = flux * 0.01

# Next, the desired output path (which may not exist yet)
out_path = 'example_generic_results'

# Finally, the path to the stacked pickle object which will be saved for later use
stack_path = os.path.join(out_path, 'stacked_data.pkl')

# Now, the line and wavelength we are interested in performing tests on
line_name = 'FeVII'
line = 6087

# Make the stack by looping over each individual spectrum and appending it
if not os.path.exists(stack_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    stack = bf.Stack()
    for i in range(flux.shape[1]):
        spec_i = bf.Spectrum(wave, flux[:, i], error[:, i], redshift=z, name=f'Spectrum {i+1}')
        stack.add_spec(spec_i)
    # Stack all the spectra and plot the results
    stack()
    stack.plot_stacked(os.path.join(out_path, 'stacked_plot'))
else:
    # If it does already exist, just load the pickle file
    stack = pickle.load(open(stack_path, 'rb'))

# Calculate the signal-to-noise ratio of each spectrum in the stack at the line of interest
for i in range(len(stack)):
    stack[i].calc_line_snr((line-10, line+10), 'snr_'+str(line))

# Calculate flux line ratios for FeVII 6087
_wl, _wr, fluxr_dict, info_dict = stack.calc_line_flux_ratios(line, dw=10, save=True, conf=None, path=out_path)
# Take the results and turn them into plots and output files
ss = stack.line_flux_report(fluxr_dict, line=line, dw=10, norm_dw=(_wl, _wr), path=out_path, plot_backend='plotly', ylim=(0, 2),
                            title_text_conf=None, title_text_snr='snr_'+str(line), plot_spec='all', conf_dict=None,
                            conf_target=0.7, inspect=None)

# Save the results
stack.save_pickle(stack_path)
np.savetxt(os.path.join(out_path, 'detections.txt'), np.array(ss).T, fmt='%s')
