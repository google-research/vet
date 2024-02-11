for d in 0.2 0.3 0.37 0.39 0.4
do
	python parameterized_sample.py --generator conv_abuse_distr_gen --n_items 840 --k_responses=4 --num_samples=1000 --distortion=${d} --exp_dir=
	python response_resampler.py --config_file=acl2024_config.csv --n_items 840 --k_responses=4 --exp_dir= --input_response_file=responses_simulated_distr_dist=${d}_gen_N=840_K=4_num_samples=1000.json
done
