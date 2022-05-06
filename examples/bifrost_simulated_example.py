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
# First, the randomized data parameters
# Make a sample where half of the spectra have no line, and half do have a line, with all other parameters being identical

snr = 10

# Make two sets of identical RNG seeds
rng_seeds = np.random.randint(10000, 99999, 100)
# rng_seeds = np.concatenate((rng_seeds, rng_seeds))

# Half with amplitudes = 0 and the other half random
# amps = np.concatenate((np.zeros((100,)), np.random.uniform(0.01, 1, 100)))  # in 10^-17 erg/s/cm^2/A
amps = np.random.uniform(0.01, 2.5, (100,3))
baselines = np.ones(100)                                                    # in 10^-17 erg/s/cm^2/A
fwhms = np.random.normal(0, 100, (100,3))                                       # in km/s
# fwhms = np.concatenate((fwhms, fwhms))
voffs = np.random.normal(0, 100, (100,3))                                       # in km/s
# voffs = np.concatenate((voffs, voffs))
noise_amplitudes = np.random.uniform(0.01, 0.3, 100)                        # in 10^-17 erg/s/cm^2/A
# noise_amplitudes = np.concatenate((noise_amplitudes, noise_amplitudes))

h_moments = np.random.uniform(0, 0, 600).reshape((100, 3, 2))                # unitless
# h_moments = np.concatenate((h_moments, h_moments))
eta_mixes = np.random.uniform(0, 1, (100,3))                                    # unitless
# eta_mixes = np.concatenate((eta_mixes, eta_mixes))
profiles = ['random', 'random', 'random']
# names = np.concatenate((np.array([f"{profiles}_{i}_noamp" for i in range(100)]),
                        # np.array([f"{profiles}_{i}_amp" for i in range(100)])))

# Next, the desired output path (which may not exist yet)
out_path = 'example_simulated_results'

# Finally, the path to the stacked pickle object which will be saved for later use
stack_path = os.path.join(out_path, 'stacked_data.pkl')

# Now, the line and wavelength we are interested in performing tests on
line_name = ['line1', 'line2', 'line3']
line = [6000, 6200, 6400]

# Make the stack using the handy classmethod, but only if it doesn't already exist:
if not os.path.exists(stack_path):
    stack = bf.Stack.quick_sim_stack(line, baselines, amps, fwhms, voffs, h_moments=h_moments, eta_mixes=eta_mixes,
                                     noise_amplitudes=noise_amplitudes, profiles=profiles,
                                     seeds=rng_seeds, names=None, out_path=out_path)
    # Stack all the spectra and plot the results
    stack()
    stack.plot_stacked(os.path.join(out_path, 'stacked_plot'))
else:
    # If it does already exist, just load the pickle file
    stack = pickle.load(open(stack_path, 'rb'))

# Calculate the signal-to-noise ratio of each spectrum in the stack at the line of interest
for i in range(len(stack)):
    stack[i].calc_line_snr((line[1]-10, line[1]+10), 'snr_'+str(line[1])+'_1')
    stack[i].data['snr_'+str(line[1])+'_2'] = amps[i, 1] / noise_amplitudes[i]

# Calculate flux line ratios for FeVII 6087
_wl, _wr, fluxr_dict, info_dict = stack.calc_line_flux_ratios(line[1], dw=10, save=True, conf=None, path=out_path)
# Take the results and turn them into plots and output files
ss = stack.line_flux_report(fluxr_dict, line=line[1], dw=10, norm_dw=(_wl, _wr), path=out_path, plot_backend='pyplot', ylim=(0, 2),
                            plot_range=(line[0]-50, line[-1]+50),
                            title_text_conf=None, title_text_snr='snr_'+str(line[1]), plot_spec='all', conf_dict=None,
                            inspect=None)
# stack.plot_spectra(out_path+'/plots/', ylim=(0,2), _range=(line[0]-50, line[-1]+50), backend='pyplot',
                   # title_text={key: "SNR1: " + str(stack[key].data['snr_'+str(line[1])+'_1']) + " SNR2: " + str(stack[key].data['snr_'+str(line[1])+'_2']) for key in stack})

# Save the results
stack.save_pickle(stack_path)
np.savetxt(os.path.join(out_path, 'detections.txt'), np.array(ss).T, fmt='%s')
