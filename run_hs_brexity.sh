for d in 0.005 0.01 0.02 0.1
do
	python parameterized_sample.py --generator hs_brexit_distr_gen --n_items 168 --k_responses=5 --num_samples=1000 --distortion=${d} --exp_dir=
	python response_resampler.py --config_file=acl2024_config.csv --n_items 168 --k_responses=5 --exp_dir= --input_response_file=responses_simulated_distr_dist=${d}_gen_N=168_K=5_num_samples=1000.json
done
