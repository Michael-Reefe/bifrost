from multiprocessing.sharedctypes import Value
from textwrap import fill
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import toml
import os
import numexpr as ne
import plotly.subplots as psp
import plotly.graph_objects as pgo

from scipy.stats import reciprocal
from skopt.searchcv import BayesSearchCV
from skopt.plots import plot_objective, plot_histogram
from skopt.space import Real, Integer, Categorical

import bifrost.maths as maths
import bifrost


class NeuralNet:

    # A class for creating and training neural networks to detect optical coronal lines

    __slots__ = ['min_wave', 'max_wave', 'spec_size', 'n_extra_params', 'loss', 'optimizer', 'metrics', 'model', 
                 'regressor', 'search_model']
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "neural.net.toml"))
    config = toml.load(config_path)

    def __init__(self, spec_size=None, min_wave=None, max_wave=None, loss=tf.keras.losses.BinaryCrossentropy,
                 optimizer=tf.keras.optimizers.Adam, metrics=('binary_accuracy',)):
        """
        Initialize the neural network.

        :param spec_size: int
            Number of pixels that the logarithmically-spaced wavelength grid for the neural network should take up,
            defaults to 101.
        :param min_wave: float
            Minimum wavelength of the grid, defaults to line-50.
        :param max_wave: float
            Maximum wavelength of the grid, defaults to line+50.
        :param loss: tensorflow.keras.losses
            Loss function for the neural network. Defaults to BinaryCrossentropy.
        :param optimizer: tensorflow.keras.optimizers
            Optimizer for the neural network.  Defaults to Adam.
        :param metrics: iterable
            Metrics for the neural network.  Defaults to ('binary_accuracy',).
        :return self:
            The NeuralNet instance.
        """

        param_distribs = {
            # Minimum of 3 dense layers (not counting the input and activation layers)
            "n_layers": Integer(3, 6, prior="uniform"),
            "n_neurons": Integer(0, 100, prior="uniform"),
            "learning_rate": Real(1e-10, 1e-1, prior="log-uniform")
        }

        self.regressor = tf.keras.wrappers.scikit_learn.KerasRegressor(self._build_model)
        self.search_model = BayesSearchCV(self.regressor, param_distribs, n_iter=50, cv=3, n_jobs=3, refit=True, verbose=9)
        self.model = None

        # Create the overall wavelength grid and the keras model
        self.min_wave = min_wave
        self.max_wave = max_wave
        self.spec_size = spec_size
        # Number of extra parameters to add to input data aside from flux:
        # 1. RMS of the flux
        self.n_extra_params = 0
        # Model hyperparameters
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
    
    def _build_model(self, n_layers, n_neurons, learning_rate):
        """
        Helper function to build a keras model with the given parameters

        :param n_layers: int
            Number of dense layers of the neural network
        :param n_neurons: int
            Number of neurons per dense layer of the neural network
        :param learning_rate: float
            The learning rate of the neural network
        :return model: tensorflow.keras.models.Sequential
            The keras neural network.
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(self.spec_size+self.n_extra_params,)))
        for layer in range(n_layers):
            model.add(tf.keras.layers.Dense(n_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss=self.loss(from_logits=True), optimizer=self.optimizer(learning_rate=learning_rate), metrics=self.metrics)
        return model
    
    def _get_wave_limits(self, line, target_line=0, size=100_000):
        training_parameters = NeuralNet.config["training_parameters"]

        # Get shape for outputs based on wavelength
        if type(line) in (str, np.str_):
            if line not in training_parameters["line_sets"].keys():
                wavelength = [training_parameters[line]['wavelength']]
                line = [line]
                SHAPE = (size,1)
            else:
                line = [line]
                wavelength = [training_parameters[line[0]]['wavelength']]
                for li in training_parameters["line_sets"][line[0]]:
                    line.append(li)
                    wavelength.append(training_parameters[li]['wavelength'])
                SHAPE = (size,len(line))
        elif type(line) in (list, np.ndarray):
            for li in line:
                if li in training_parameters["line_sets"]:
                    for lli in training_parameters["line_sets"][li]:
                        if lli not in line:
                            line.append(lli)
            wavelength = [training_parameters[li]['wavelength'] for li in line]
            SHAPE = (size,len(line))
        else:
            raise ValueError("line must be one of: int, float, list, np.ndarray")
        line = np.asarray(line)
        wavelength = np.asarray(wavelength)

        if self.min_wave is None:
            self.min_wave = np.min(wavelength) - 50
        if self.max_wave is None:
            self.max_wave = np.max(wavelength) + 50
        if self.spec_size is None:
            self.spec_size = int(np.max(wavelength) - np.min(wavelength) + 101)
        
        return line, wavelength, SHAPE


    def train(self, line="generic_line", target_line=0, size=100_000, epochs=11, out_path="neuralnet_training_data", save=True,
              save_path="bifrost.neuralnet.h5", plot=True):
        """
        The main function for training the neural network to recognize a specific optical coronal line ("line") using simulated data

        :param line: str, iterable
            The line name(s) to include in the simulated training data
        :param target_line: int
            The index of which line in the "line" argument is the line to train the neural network to detect. Default = 0.
        :param size: int
            The size of the simulated dataset. Default = 100_000.
        :param epochs: int
            The number of epochs to train the data for. Default = 11.
        :param out_path: str
            The output path for saving the neural network h5 file, or any plots. Default = 'neuralnet_training_data'
        :param save: bool
            Whether or not to save the trained network as an h5 file. Default = True.
        :param save_path: str
            The name of the h5 file to be saved if save=True. Default = 'bifrost.neuralnet.h5'
        :param plot: bool
            Whether or not to plot a subset of the simulated training data (100). Default = True.
        :return None:
        """

        training_parameters = NeuralNet.config["training_parameters"]
        # Make RNG seeds
        power = int(np.log10(size))
        rng_seeds = np.random.randint(int(10**power), int(10**(power+1)-1), size)

        line, wavelength, SHAPE = self._get_wave_limits(line, target_line, size)

        # Initialize output parameters, baselines, and noise
        output_parameters = {}
        baselines = np.ones(size)
        constrained = []               

        # Initialize parameters that are constant for every line: random noise and the power slope
        for param in ("noise", "power_slope"):
            if training_parameters[param]['dist'] == 'uniform':
                p1, p2 = 'min', 'max'
            elif training_parameters[param]['dist'] in ('normal', 'lognormal'):
                p1, p2 = 'mean', 'std'
            else:
                raise ValueError(f"Unrecognized dist config parameter '{training_parameters[param]['dist']}'")
            output_parameters[param] = eval(f"np.random.{training_parameters[param]['dist']}({training_parameters[param][p1]}," 
                        f"{training_parameters[param][p2]}, {size})")

        # Initialize parameters that are different for every line
        for param in ("amp", "fwhm", "voff", "h3", "h4", "eta"):
            output_parameters[param] = np.zeros(SHAPE)
            # Create the distributions
            output_parameters[param] = np.zeros(SHAPE)
            for i, li in enumerate(line):
                # Check for profile-specific features
                if param in ("h3", "h4") and training_parameters[li]["profile"] not in ("GH", "random"):
                    output_parameters[param][:, i] = np.nan
                    continue
                if param == "eta" and training_parameters[li]["profile"] not in ("V", "random"):
                    output_parameters[param][:, i] = np.nan
                    continue

                # Generate keywords
                if training_parameters[li][param]['dist'] == 'uniform':
                    p1, p2 = 'min', 'max'
                elif training_parameters[li][param]['dist'] in ('normal', 'lognormal'):
                    p1, p2 = 'mean', 'std'
                elif training_parameters[li][param]['dist'] == 'constrained':
                    constrained.append((i, param))
                    continue
                else:
                    raise ValueError(f"Unrecognized dist config parameter '{training_parameters[li][param]['dist']}'")

                output_parameters[param][:, i] = eval(f"np.random.{training_parameters[li][param]['dist']}"
                    f"({training_parameters[li][param][p1]}, {training_parameters[li][param][p2]}, {size})")
        
        # Make some ratio of the simulated spectra have no line of interest
        # output_parameters["amp"][:output_parameters["amp"].shape[0]//2, target_line] = 0.
        for i in range(len(line)):
            if type(training_parameters[line[i]]["detect_ratio"]) is float:
                # Small adjustment to the ratio to make the end result be more close to the input ratio 
                # (accounting for those with SNR < the threshold)
                epsilon = 1.2 if training_parameters[line[i]]["detect_ratio"] < 0.833 else 1

                idx = int((1 - epsilon*training_parameters[line[i]]["detect_ratio"]) * len(output_parameters["amp"][:, i]))
                ids = np.random.choice(range(len(output_parameters["amp"][:, i])), idx, replace=False)
                output_parameters["amp"][ids, i] = 0.
        # Need to make 2 passes in case the line that doublets depend on appears after it does in the line list
        for i in range(len(line)):
            if type(training_parameters[line[i]]["detect_ratio"]) is str: 
                li = np.where(line == training_parameters[line[i]]["detect_ratio"])[0][0]
                noline = np.where(output_parameters["amp"][:, li] == 0.)[0]
                output_parameters["amp"][noline, i] = 0.
        # Special case for [Fe X] -- remove any detections where [O I] is not detected
        if "FeX_6374" in line and "OI_6302" in line and "OI_6365" in line:
            oi1 = np.where(line == "OI_6302")[0][0]
            fi = np.where(line == "FeX_6374")[0][0]
            noline = np.where(output_parameters["amp"][:, oi1] == 0.)[0]
            output_parameters["amp"][noline, fi] = 0.
            # yesline = np.where(output_parameters["amp"][:, oi1] > 0.)[0]
            # idy = int(training_parameters[line[fi]]["detect_ratio"] * len(yesline))
            # output_parameters["amp"][yesline[:idy], fi] = 0.

        for c in constrained:
            constraint = training_parameters[line if type(line) in (str, np.str_) else line[c[0]]][c[1]]
            assert constraint['dist'] == 'constrained'
            l_expr, n_expr = constraint['constraint'][0], constraint['constraint'][1]

            # Aliasing for nice-looking numexpr strings in the toml file:
            c_l = 0 if type(line) in (str, np.str_) else np.where(line == l_expr)[0][0]
            amp = output_parameters["amp"][:, c_l]
            noise = output_parameters["noise"][c_l]
            fwhm = output_parameters["fwhm"][:, c_l]
            voff = output_parameters["voff"][:, c_l]
            h3 = output_parameters["h3"][:, c_l]
            h4 = output_parameters["h4"][:, c_l]
            eta = output_parameters["eta"][:, c_l]
            # Randomized coefficients for use in numexpr
            randc = np.random.uniform(0.7, 1, len(amp))
            randc2 = np.random.uniform(0.3, 0.5, len(amp))
        
            output_parameters[c[1]][:, c[0]] = ne.evaluate(n_expr)

        # Parameter constraints
        # Ensure certain parameters remain positive
        # output_parameters["amp"][output_parameters["amp"] < 0] = 0
        if output_parameters["eta"] is not None:
            output_parameters["eta"][output_parameters["eta"] < 0] = 0
            output_parameters["eta"][output_parameters["eta"] > 1] = 1
        # Ensure certain parameters remain nonzero
        output_parameters["fwhm"][output_parameters["fwhm"] <= 0] = 1
        output_parameters["noise"][output_parameters["noise"] <= 0] = 0.01


        profile = [training_parameters[li]["profile"] for li in line]
        h_moments = np.dstack((output_parameters["h3"], output_parameters["h4"]))

        print("Preparing data for neural network training...")
        # breakpoint()
        # Create simulated data
        stack = bifrost.Stack.quick_sim_stack(
            wavelength, baselines, output_parameters["amp"], output_parameters["fwhm"],
            output_parameters["voff"], output_parameters["power_slope"], h_moments=h_moments, eta_mixes=output_parameters["eta"],
            noise_amplitudes=output_parameters["noise"], profiles=profile, min_wave=self.min_wave,
            max_wave=self.max_wave, size=self.spec_size, seeds=rng_seeds, out_path=out_path, progress_bar=True)

        labels = np.where(np.abs(output_parameters["amp"][:, target_line] / output_parameters["noise"]) > NeuralNet.config["SNR"], 1, 0)
        # labels = np.where(output_parameters["amp"][:, target_line] > 0, 1, 0)
        # labels = np.concatenate(([0] * (size // 2), [1] * (size // 2)))
        # labels = np.array([(0,1)[stack[i].data["snr"][target_line] >= NeuralNet.config["SNR"]] for i in range(len(stack))])
        # For debugging purposes:
        if plot:
            stack.plot_spectra(out_path, backend='pyplot', spectra=list(stack.keys())[0:100], 
                            _range=(self.min_wave, self.max_wave), normalized=True,
                            ylim=(-5,5), title_text={label: f"Line: {labels[i]}, SNR: {output_parameters['amp'][i, target_line]/output_parameters['noise'][i]:.2f}, "
                            f"Amp: {output_parameters['amp'][i, target_line]:.2f}, Noise: {output_parameters['noise'][i]:.2f}" for i, label in enumerate(stack)})

        # Split data into train, validation, and test sets
        ind = np.array(range(len(stack)))
        np.random.shuffle(ind)
        i1 = len(stack)*8//10
        i2 = len(stack)*9//10
        train_set = ind[0:i1]
        valid_set = ind[i1:i2]
        test_set = ind[i2:]

        # Resample each spectrum onto full_wave_grid, preserving flux and error
        train_data, tmu, tsig = self.normalize(np.array([stack[i].flux for i in train_set], dtype=float))
        # train_err = np.array([np.nanstd(stack[i].flux) for i in train_set], dtype=float).reshape((len(train_data), 1))
        # train_data = np.hstack((train_data, train_err))
        train_labels = np.array([labels[i] for i in train_set], dtype=int)

        valid_data, vmu, vsig = self.normalize(np.array([stack[i].flux for i in valid_set], dtype=float))
        # valid_err = np.array([np.nanstd(stack[i].flux) for i in valid_set], dtype=float).reshape((len(valid_data), 1))
        # valid_data = np.hstack((valid_data, valid_err))
        valid_labels = np.array([labels[i] for i in valid_set], dtype=int)

        test_data, ttmu, ttsig = self.normalize(np.array([stack[i].flux for i in test_set], dtype=float))
        # test_err = np.array([np.nanstd(stack[i].flux) for i in test_set], dtype=float).reshape((len(test_data), 1))
        # test_data = np.hstack((test_data, test_err))
        test_labels = np.array([labels[i] for i in test_set], dtype=int)

        self.search_model.fit(train_data, train_labels, epochs=epochs, validation_data=(valid_data, valid_labels))

        self.model = self.search_model.best_estimator_.model
        self.model.evaluate(test_data, test_labels, verbose=2)
        if save:
            self.model.save(os.path.join(out_path, save_path))
        if plot:
            plot_objective(self.search_model.optimizer_results_[0], dimensions=["learning_rate", "n_layers", "n_neurons"],
                           n_minimum_search=int(1e8))
            plt.savefig(os.path.join(out_path, "modelsearch.cornerplot.pdf"), dpi=300, bbox_inches="tight")
            plt.close()
            for i in range(3):
                plot_histogram(self.search_model.optimizer_results_[0], i)
                plt.savefig(os.path.join(out_path, f"modelsearch.histogram{i}.pdf"), dpi=300, bbox_inches="tight")
                plt.close()

    @staticmethod
    def normalize(data):
        """
        A helper function to normalize training data to have a mean of 0 and standard deviation of 1.

        :param data: np.ndarray
            The array of data to be normalized.
        :return: np.ndarray
            The normalized data.
        """
        mu = np.nanmean(data, axis=-1)
        sigma = np.nanstd(data, axis=-1)
        return ((data.T - mu) / sigma).T, mu, sigma

    def load(self, path, line, target_line=0):
        """
        Load an already-trained keras model into the neural network via an h5 file.

        :param path: str
            The path to the h5 file to load.
        :return None:
        """
        self._get_wave_limits(line, target_line, size=100_000)
        self.model = tf.keras.models.load_model(path)

    def predict(self, test_stack, p_layer="sigmoid", plot=False, out_path=None):
        """
        Use the trained neural network to predict the presence of coronal lines in real data.

        :param test_stack: bifrost.Stack
            A Stack object containing the spectra to test for coronal lines.
        :param p_layer: str
            The type of activation layer to use for the prediction results. Default = 'sigmoid'.
        :param plot: bool
            Whether or not to plot the resultant spectra with their predictions. Default = False.
        :param out_path: str
            The output path to save plots if plot = True.
        :return predictions: np.ndarray
            Array of prediction values (confidences between 0-1) for each spectrum in the input stack.
        """
        # Construct wavelength grid for the inputs
        wave_grid = np.geomspace(self.min_wave, self.max_wave, self.spec_size)
        # Correct for redshift and galactic extinction before testing
        test_stack.correct_spectra()
        # Resample input spectra onto the proper shaped wavelength grid for input to the neural network
        test_data, ttmu, ttsig = self.normalize(np.array([
            maths.spectres(wave_grid, test_stack[i].wave, test_stack[i].flux, test_stack[i].error, fill=np.nan)[0]
            for i in range(len(test_stack))], dtype=float))

        # test_data, ttmu, ttsig = self.normalize(np.array([test_stack[i].flux for i in range(len(test_stack))], dtype=float))
        # test_err = ((np.array([test_stack[i].error for i in range(len(test_stack))], dtype=float).T - ttmu) / ttsig).T

        # test_err = np.array([np.nanstd(test_stack[i].flux) for i in range(len(test_stack))], dtype=float).reshape((len(test_data), 1))
        # test_data = np.hstack((test_data, test_err))

        # Make sure there are no nans or infs so that the neural net still works
        for i in range(test_data.shape[0]):
            if np.isfinite(np.nanmedian(test_data[i, :])):
                test_data[i, ~np.isfinite(test_data[i, :])] = np.nanmedian(test_data[i, :])
            else:
                test_data[i, :] = 0.

        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Activation(p_layer)])
        predictions = probability_model.predict(test_data)
        print(predictions)

        if plot:
            if out_path is None:
                out_path = "neuralnet_training_data"
            test_stack.plot_spectra(out_path, backend='pyplot',
                _range=(self.min_wave, self.max_wave), ylim=(-5,5),
                spectra=np.asarray(list(test_stack.keys()))[np.where(predictions > 0.99)[0]],  # only plot the really confident spectra
                title_text={label: f"NN Confidence: {predictions[k]}" for k, label in enumerate(test_stack.keys())},
                normalized=True)

        return predictions
    
    def convolve(self, wave, flux, error, p_layer="sigmoid", plot=True, out_path=None):
        # Take the neural network as a sliding window along the length of the spectrum,
        # generating a confidence as a function of wavelength

        # Gather the left and right wavelength bounds for each window
        slide_size = len(wave)-self.spec_size+1
        left_wave = np.geomspace(wave[0], wave[-self.spec_size], slide_size)
        right_wave = np.geomspace(wave[self.spec_size-1], wave[-1], slide_size)

        cwave = np.full(slide_size, fill_value=np.nan)
        data = np.full((slide_size, self.spec_size), fill_value=np.nan)

        # Go through and resample the flux for each window onto the proper wavelength grid
        for i in range(slide_size):
            wgi = np.geomspace(left_wave[i], right_wave[i], self.spec_size)
            cwave[i] = np.nanmedian(wgi)
            data[i, :], _, _ = self.normalize(maths.spectres(wgi, wave, flux, error, fill=np.nan)[0])
            if np.isfinite(np.nanmedian(data[i, :])):
                data[i, ~np.isfinite(data[i, :])] = np.nanmedian(data[i, :])
            else:
                data[i, :] = 0.
        
        # Perform the neural network predictions across each window
        probability_model = tf.keras.Sequential([self.model, tf.keras.layers.Activation(p_layer)])
        conf = probability_model.predict(data).T[0]
        cflux, _, _ = self.normalize(maths.spectres(cwave, wave, flux, error, fill=np.nan)[0])

        # Plot the probability vs. wavelength
        if plot:
            if out_path is None:
                out_path = "neuralnet_training_data/nn.convolve.confidence.html"
            # fig, ax = plt.subplots(figsize=(10,5))
            # ax.plot(cwave, conf, 'k-')
            # ax.set_xlabel(r'$\lambda (\AA)$')
            # ax.set_ylabel(r'Line Confidence')
            # plt.savefig(os.path.join(out_path, "nn.convolve.confidence.pdf"), dpi=300, bbox_inches='tight')
            # plt.close()
            fig = psp.make_subplots(rows=1, cols=1)
            fig.add_trace(pgo.Scatter(x=cwave, y=conf, line=dict(color='red', width=2), name='Confidence', showlegend=False))
            fig.add_trace(pgo.Scatter(x=cwave, y=cflux, line=dict(color='black', width=1), name='Flux', showlegend=False))
            fig.update_layout(
                xaxis_title=r'$\lambda (A)$',
                yaxis_title=r'Confidence | Flux (normalized)',
                template='plotly_white'
            )
            fig.write_html(out_path, include_mathjax="cdn")

        return cwave, conf