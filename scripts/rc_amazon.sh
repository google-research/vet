n_items=20415
k_responses=5
for d in 0.005 0.01 0.02 0.1
do
	sbatch scripts/empirical_sample.sh ${d} ${n_items} ${k_responses} amazon_distr_gen
done
