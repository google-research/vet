for d in 0.1 0.2 0.3 0.37 0.4
do
	python parameterized_sample.py --generator hs_brexit_distr_gen --n_items 168 --k_responses=6 --num_samples=1000 --distortion=${d} --exp_dir=
	python response_resampler.py --config_file=acl2024_config.csv --n_items 168 --k_responses=6 --exp_dir= --input_response_file=responses_simulated_distr_dist=${d}_gen_N=168_K=6_num_samples=1000.json
done
python gather_table.py --n_items=168 --k_responses=6 --distortion=0.1,0.2,0.3,0.37,0.4