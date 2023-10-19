# P-value estimator

## Overview
This repository contains code for testing the accuracy of p-value estimators for
machine learning contests. 
It has three main components:

   1. `parameterized_sample.py` Generates a large (typically 1000) number of
   samples of simulated output from three sources: a
   pool of human annotators and two machines, using known probability
   distributions.
   2. `response_resampler.py` Estimates p-values directly on the data generated
   by `parameterized_sample.py` and also via resampling on the first sampled set
   of data from `parameterized_sample.py` It then compares the results on the
   generated data and resampled data, where the resampled estimates are the
   kinds of estimates available to investigators under normal experiment
   conditions, and the estimates from `parameterized_sample.py` are a more
   accurate value of the p-value under a null hypothesis based on the actual
   distributions that generated the data. `response_resampler.py` can be run
   in parallel, to save time.
   3. `reconstruct_borg_job.py` is used when `response_resampler.py` is run in
   parallel, to collect the outputs of each job into a single output.

## Usage
Example usage:

python parameterized_sample --path_name=/data_dir/path --distortion=.02

Where:
--path_name is the name of the path where the data files for this program
reside. By default it is the current directory.

--distortion A floating point number controlling the amount of error in the
second machine relative to the human response distribution. This is used with
--generator=alt_distr_gen argument, but not for all other distribution
generators.

--input Used for data
_GENERATOR = flags.DEFINE_string(
    "generator", "alt_distr_gen", "A function that generates distributions."
)

_N_ITEMS = flags.DEFINE_integer(
    "n_items", 1000, "Number of items per response set."
)
_K_RESPONSES = flags.DEFINE_integer(
    "k_responses", 5, "Number of responses per item."
)


python response_resampler --

reconstruct --path=/path/to/output/ --file_name=output_file_prefix --lines=45

Where:
--path is the name of the path to the output files for the parallel runs
of response_resampler.py
--file_name is the name of the prefix (everything before "=x.csv" in the file
name, where x is an integer corresponding to a parallel run of
response_resampler.py
--lines is the number of parallel runs to reconstruct.
Each run produces a one-line csv file
