n_items=840
k_responses=4
for d in 0.1 0.12 0.16 0.2
do
	sbatch scripts/empirical_sample.sh ${d} ${n_items} ${k_responses} convabuse_gen
done
