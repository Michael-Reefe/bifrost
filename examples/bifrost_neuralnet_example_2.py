# First import some base modules
import os
import pickle
import sys
# Import numpy
import numpy as np
# Import the bifrost module
# NOTE: this WILL NOT WORK until you have installed the module with `pip install .` from the terminal.
import bifrost as bf

# The coronal line name that should be tested
line_name = ['metal_abs', 'generic_line']

# Construct and train the neural network to find detections for the line in question
nn = bf.NeuralNet()
# nn.train(line_name, target_line=0, size=100_000, epochs=11)
nn.load('neuralnet_training_data/bifrost.neuralnet.h5', line_name, target_line=0)

for path in ['example_data/spec-7721-57360-0412.fits', 'example_data/spec-7673-57329-0248.fits']:
    test_spec = bf.Spectrum.from_fits(path, None)
    test_spec.apply_corrections()
    wave = test_spec.wave
    flux = test_spec.flux
    error = test_spec.error

    cwave, conf = nn.convolve(wave, flux, error, out_path=f"nn.convolve.{path.split(os.sep)[1].split('.')[0]}.html")

# Test how well the neural network works on real data
# test_stack = bf.Stack.quick_fits_stack('example_data', out_path='neuralnet_training_data')
# test_stack.correct_spectra()

# nn.predict(test_stack, plot=True)