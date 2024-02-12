n_items=145
k_responses=3
for d in 0.37 0.49 0.53 0.57 0.61
do
	python parameterized_sample.py --generator armis_distr_gen --n_items=${n_items} --k_responses=${k_responses} --num_samples=1000 --distortion=${d} --exp_dir=
	python response_resampler.py --config_file=acl2024_config.csv --n_items=${n_items} --k_responses=${k_responses} --exp_dir= --input_response_file=responses_simulated_distr_dist=${d}_gen_N=${n_items}_K=${k_responses}_num_samples=1000.json
done
python gather_table.py --n_items=${n_items} --k_responses=${k_responses} --distortion=0.37,0.49,0.53,0.57,0.61