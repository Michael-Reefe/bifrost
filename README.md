# BIFR&#xd6;ST
## Black hole Investigations For energetic Radiation On Spectra via STacking

This code aims to bridge the gap between stellar mass black holes and supermassive
black holes, just as the burning rainbow bridge it is named after bridges the gap
between Asgard and Midgard, by providing a means of searching for them via a
spectral stacking framework.

### Installation
After cloning the repository, you can install locally via pip: ` pip install . `
Or alternatively, ` python setup.py install `.

It may be necessary to increase the open file limit with ` ulimit -n XXXX `, depending
on how many spectra you are stacking at once.

### CLI
The code can be run via the command-line interface.  The syntax for stacking spectra is:
``` 
usage: bifrost stack [-h] [--out-path PATH] [--n-jobs N] [--no-save-pickle] [--save-json] [--pyplot] [--plot-spec N [N ...]]
                     data_path

positional arguments:
  data_path             Path to a directory with fits files.

optional arguments:
  -h, --help            show this help message and exit
  --out-path PATH, -o PATH
                        Save path for outputs.
  --n-jobs N, -n N      Number of jobs to run in parallel when reading fits files.
  --no-save-pickle, -np
                        Use this option if you do not want to save a pickle file of the stack object.
  --save-json, -js      Use this option if you want to save a json file of the stack object.
  --pyplot, -pp         Use this option to plot with the pyplot module instead of plotly.
  --plot-spec N [N ...], -ps N [N ...]
                        Specify the indices of which spectra to plot individually.
``` 
