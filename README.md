# Variance Estimation Toolkit (VET)
Disclaimer: This is not an officially supported Google product.

## Overview
This repository contains code for generating simulated item/response distributions of various shapes, and measuring the p-value of comparisons between those distributions.  Its primary use case is facilitating the power analysis of human ratings for comparing two versions of an AI system.
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

## Usage
### Example usage for response sample generation:
```shell
python parameterized_sample.py --exp_dir=/data_dir/path --distortion=.02
```

Where:

`--exp_dir` is the file path where the experiment input and output data are located.

`--generator` is the type of random sample generator. Right now it supports 
  either normal distribution generator (`ALT_DISTR_GEN`) or Likert normal distribution generator (`TOXICITY_DISTR_GEN`).

`--distortion` is a floating point number controlling the amount of distribution generation error in the
second machine sample distribution relative to the human sample distribution.

`--n_items` is the number of items per response set. Each item is assumed to have its own distribution.

`--k_responses` is the number of responses per item. They can be annotators responses for human results or machine responses for machine results.

`--num_trials` is the number of trials per experiment. Each trial contains `n_items * k_responses` samples for human, machine1 and machine2 results.

`--use_pickle` decides whether to save the sample data in pickle format. When set to false, the samples are saved in json format, which is more readable but less efficient in storage space.

### Example usage for computing metrics over the generated samples:
```shell
python response_resampler.py --exp_dir=/path/to/experiment/ --input_response_file=input_file_prefix --config_file=config.csv --line_num=45
```

Where:

`--exp_dir` is the path to the input/output files for the experiment.

`--input_response_file` is the name of the input file. The output file name is generated based on the input file name and the experiment config.

`--line_num` is the number of config line to run. When running in parallel, each resampler job can do the processing according to one line of the experiment config.

`--config_file` is the name of the config csv file located in `<exp_dir>/config/<config_file>`.

`--n_items` and `--k_responses` are similar to the flags in `parameterized_sample.py`. They can be redefined for resampling.
