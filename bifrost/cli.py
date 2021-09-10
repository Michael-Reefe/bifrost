import argparse
import sys
import copy

from bifrost.driver import *


def cli_run(args):
    """
    Passes the CLI arguments into the main driver function in driver.

    :param args : parser.parse_args
        Arguments passed in from the command line.
    :return None:
    """
    backend = 'pyplot' if args.use_pyplot else 'plotly'

    # Parse table property arguments
    prop_tbl = None
    prop_sep = ","
    prop_com = "#"
    prop_name_col = 0
    if args.props:
        prop_tbl = args.props[0]
        prop_args = len(args.props)
        if prop_args > 1:
            prop_sep = args.props[1]
        if prop_args > 2:
            prop_com = args.props[2]
        if prop_args > 3:
            prop_name_col = int(args.props[3])
        if prop_args > 4:
            raise SyntaxError("Found too many arguments for table properties!")

    # Parse binning arguments
    bin_quant = None
    bin_num = None
    bin_log = False
    if args.bin:
        bin_quant = args.bin[0]
        bin_num = int(args.bin[1])
        if 'log_' in bin_quant:
            bin_quant = bin_quant.replace('log_', '')
            bin_log = True

    driver(args.data_path, out_path=args.out_path, n_jobs=args.n_jobs, save_pickle=args.pickle, save_json=args.json,
           plot_backend=backend, plot_spec=args.plot_spec, limits=args.limit, _filters=args.filters, name_by=args.name_by,
           properties_tbl=prop_tbl, properties_comment=prop_com, properties_sep=prop_sep, properties_name_col=prop_name_col,
           bin_quant=bin_quant, nbins=bin_num, bin_size=None, bin_log=bin_log, hist_log=args.hist_log)


def plot_run(args):
    """
    Passes the CLI arguments into the plotter function in driver.
    :param args: parser.parse_args
        Arguments passed in from the command line
    :return None:
    """
    backend = 'pyplot' if args.use_pyplot else 'plotly'
    if args.hist or args.hist_log:
        hist = True
    else:
        hist = False
    plotter(args.stack_path, out_path=args.out_path, plot_backend=backend, plot_spec=args.plot_spec, plot_hist=hist,
            plot_log=args.hist_log)


def config_run(args):
    """
    Passes in the CLI arguments to the edit_config function in driver.

    :param args: parser.parse_args
        Arguments passed in from the command line
    :return None:
    """
    # Remove any arguments that are None, so they wont be changed
    args = args.__dict__
    reset = args['norm_reset']
    for arg in copy.deepcopy(args):
        if (args[arg] is None) or arg in ('func', 'norm_reset'):
            del args[arg]
    if reset:
        args['norm_region'] = None

    edit_config(**args)


def main():
    """
    Defines the 'run' CLI command and arguments.

    :return None:
    """
    parser = argparse.ArgumentParser(description='Spectra Stacking Code')
    subparsers = parser.add_subparsers()

    # Main stacking command
    run_driver = subparsers.add_parser('stack', help='Create a stacked spectrum')
    run_driver.add_argument('data_path', help='Path to a directory with fits files.')
    run_driver.add_argument('--out-path', '-o', metavar='PATH', help='Save path for outputs.', dest='out_path')
    run_driver.add_argument('--n-jobs', '-n', metavar='N', type=int, dest='n_jobs', default=-1,
                            help='Number of jobs to run in parallel when reading fits files.')
    run_driver.add_argument('--no-save-pickle', '-k', action='store_false', dest='pickle',
                            help='Use this option if you do not want to save a pickle file of the '
                                 'stack object.')
    run_driver.add_argument('--save-json', '-j', action='store_true', dest='json',
                            help='Use this option if you want to save a json file of the stack object.')
    run_driver.add_argument('--pyplot', '-p', action='store_true', dest='use_pyplot',
                            help='Use this option to plot with the pyplot module instead of plotly.')
    run_driver.add_argument('--plot-spec', '-s', metavar='N', type=int, nargs='+', dest='plot_spec', default=None,
                            help='Specify the indices of which spectra to plot individually.')
    run_driver.add_argument('--limit', '-l', metavar='N', type=int, nargs=2, dest='limit', default=None,
                            help='Limit to only use the data between 2 indices.')
    run_driver.add_argument('--filters', '-f', metavar='STR', type=str, nargs='+', dest='filters', default=None,
                            help='Add filters to the spectra used in the stack.')
    run_driver.add_argument('--name-by', '-N', metavar='STR', type=str, dest='name_by', default='folder',
                            help='Name objects by their file name, folder name, or IAU name.  If folder, cannot have 2 objects '
                                 'within the same folder.')
    run_driver.add_argument('--table', '-t', metavar='STR', type=str, nargs='+', dest='props', default=None,
                            help='File path to a table with information about each spectrum.  Names in the table must '
                                 'match the file or folder names of each spectrum.  Optional additional parameters include '
                                 'the delimiter [SEP], comment character [COM], and name column index [I], in that order.')
    run_driver.add_argument('--bin', '-b', metavar='STR', type=str, nargs=2, dest='bin', default=None,
                            help='Quantity to bin by, must be within the table file, followed by the number of bins. '
                                 'If you want to take the log10 of the item before binning, enter as "log_[item]". '
                                 'i.e. to bin by the log of a key called \'MASS_1\' with 10 bins: -b log_MASS_1 10')
    run_driver.add_argument('--bin-log', '-bl', action='store_true', dest='hist_log', help='Make the y-axis of the binned'
                                                                                           ' histogram logarithmic.')
    run_driver.set_defaults(func=cli_run)

    # Replotting command
    plot_driver = subparsers.add_parser('plot', help='Replot a stacked spectrum from a saved object')
    plot_driver.add_argument('stack_path', help='Path to a stack pickle or json file.')
    plot_driver.add_argument('--out-path', '-o', metavar='PATH', help='Save path for outputs.', dest='out_path', default=None)
    plot_driver.add_argument('--pyplot', '-p', action='store_true', dest='use_pyplot',
                             help='Use this option to plot with the pyplot module insteaad of plotly.')
    plot_driver.add_argument('--plot-spec', '-s', metavar='N', nargs='+', dest='plot_spec', default=None,
                             help='Specify the indices of which spectra to plot individually instead of plotting the stack.')
    plot_driver.add_argument('--plot-hist', '-H', action='store_true', dest='hist', help='Plot histogram of binned data.')
    plot_driver.add_argument('--plot-hist-log', '-Hl', action='store_true', dest='hist_log', help='Plot a logarithmic histogram'
                                                                                                  ' of binned data.')
    plot_driver.set_defaults(func=plot_run)

    # Edit config command
    config_driver = subparsers.add_parser('config', help='Edit default configurations for stacks.')
    config_driver.add_argument('-r_v', metavar='F', type=float, help='Extinction ratio A(V)/E(B-V) to calculate for.', dest='r_v', default=None)
    config_driver.add_argument('-gridspace', metavar='F', type=float, help='Spacing of the wavelength grid.', dest='gridspace',
                               default=None)
    config_driver.add_argument('-tolerance', metavar='F', type=float, help='Tolerance for throwing out spectra that are > tolerance angstroms apart from others.',
                               dest='tolerance', default=None)
    config_driver.add_argument('-norm-region', metavar='F', nargs=2, help='Wavelength bounds to use for the normalization region, with no prominent lines.',
                               dest='norm_region', default=None)
    config_driver.add_argument('--reset-norm-region', action='store_true', dest='norm_reset',
                               help='Reset the normalization region to None, which automatically calculates the ideal region.')
    config_driver.set_defaults(func=config_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
    sys.exit()
