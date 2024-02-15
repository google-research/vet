#!/bin/bash -l
conda activate vet
python3 /home/cmhvcs/vet/parameterized_sample.py --generator alt_distr_gen --n_items 10 --k_responses 10 --num_samples 1000 --distortion .1 --exp_dir /home/cmhvcs/vet/data
