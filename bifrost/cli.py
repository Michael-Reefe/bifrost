import argparse
import sys

from bifrost.driver import driver


def cli_run(args):
    """
    Passes the CLI arguments into the run function in driver.

    :param args : parser.parse_args
        Arguments passed in from the command line.
    :return None:
    """
    backend = 'pyplot' if args.use_pyplot else 'plotly'
    driver(args.data_path, out_path=args.out_path, n_jobs=args.n_jobs, save_pickle=args.pickle, save_json=args.json,
           plot_backend=backend, plot_spec=args.plot_spec, limits=args.limit)


def main():
    """
    Defines the 'run' CLI command and arguments.

    :return None:
    """
    parser = argparse.ArgumentParser(description='Spectra Stacking Code')
    subparsers = parser.add_subparsers()
    run_driver = subparsers.add_parser('stack', help='Create a stacked spectrum')
    run_driver.add_argument('data_path', help='Path to a directory with fits files.')
    run_driver.add_argument('--out-path', '-o', metavar='PATH', help='Save path for outputs.', dest='out_path')
    run_driver.add_argument('--n-jobs', '-n', metavar='N', type=int, dest='n_jobs', default=-1,
                            help='Number of jobs to run in parallel when reading fits files.')
    run_driver.add_argument('--no-save-pickle', '-np', action='store_false', dest='pickle',
                            help='Use this option if you do not want to save a pickle file of the '
                                 'stack object.')
    run_driver.add_argument('--save-json', '-js', action='store_true', dest='json',
                            help='Use this option if you want to save a json file of the stack object.')
    run_driver.add_argument('--pyplot', '-pp', action='store_true', dest='use_pyplot',
                            help='Use this option to plot with the pyplot module instead of plotly.')
    run_driver.add_argument('--plot-spec', '-ps', metavar='N', type=int, nargs='+', dest='plot_spec', default=None,
                            help='Specify the indices of which spectra to plot individually.')
    run_driver.add_argument('--limit', '-l', metavar='A B', type=int, nargs=2, dest='limit', default=None,
                            help='Limit to only use the data between A and B indices.')
    run_driver.set_defaults(func=cli_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
    sys.exit()
