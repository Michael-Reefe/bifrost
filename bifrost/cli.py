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
    driver(args.data_path, out_path=args.out_path, n_jobs=args.n_jobs, save_pickle=args.pickle, save_json=args.json,
           plot_backend=backend, plot_spec=args.plot_spec, limits=args.limit, _filters=args.filters, name_by=args.name_by)


def plot_run(args):
    """
    Passes the CLI arguments into the plotter function in driver.
    :param args: parser.parse_args
        Arguments passed in from the command line
    :return None:
    """
    backend = 'pyplot' if args.use_pyplot else 'plotly'
    plotter(args.stack_path, out_path=args.out_path, plot_backend=backend, plot_spec=args.plot_spec)


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
    run_driver.add_argument('--name-by', '-b', metavar='STR', type=str, dest='name_by', default='folder',
                            help='Name objects by their file name or folder name.  If folder, cannot have 2 objects '
                                 'within the same folder.')
    run_driver.set_defaults(func=cli_run)

    # Replotting command
    plot_driver = subparsers.add_parser('plot', help='Replot a stacked spectrum from a saved object')
    plot_driver.add_argument('stack_path', help='Path to a stack pickle or json file.')
    plot_driver.add_argument('--out-path', '-o', metavar='PATH', help='Save path for outputs.', dest='out_path', default=None)
    plot_driver.add_argument('--pyplot', '-p', action='store_true', dest='use_pyplot',
                             help='Use this option to plot with the pyplot module insteaad of plotly.')
    plot_driver.add_argument('--plot-spec', '-s', metavar='N', nargs='+', dest='plot_spec', default=None,
                             help='Specify the indices of which spectra to plot individually instead of plotting the stack.')
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
