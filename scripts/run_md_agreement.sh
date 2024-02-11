for d in 0.2 0.3 0.37 0.39 0.4
do
	python parameterized_sample.py --generator md_agreeement_gen --n_items 3057 --k_responses=5 --num_samples=1000 --distortion=${d} --exp_dir=
	python response_resampler.py --config_file=acl2024_config.csv --n_items 3057 --k_responses=5 --exp_dir= --input_response_file=responses_simulated_distr_dist=${d}_gen_N=3057_K=5_num_samples=1000.json
done
python gather_table.py --n_items=30 --k_responses=5 --distortion=0.2,0.3,0.37,0.39,0.4