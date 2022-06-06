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
line_name = 'FeX_6374'

# Construct and train the neural network to find detections for the line in question
nn = bf.NeuralNet()
nn.train(line_name, target_line=0, size=100_000, epochs=11)
# nn.load('neuralnet_training_data/bifrost.neuralnet.h5')

# Test how well the neural network works on real data
test_stack = bf.Stack.quick_fits_stack('example_data', out_path='neuralnet_training_data')
test_stack.correct_spectra()

nn.predict(test_stack, plot=True)