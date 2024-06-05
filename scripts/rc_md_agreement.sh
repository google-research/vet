n_items=3057
k_responses=5
for d in 0.12 0.14 0.16 0.18 0.19 
do	
	sbatch scripts/empirical_sample.sh ${d} ${n_items} ${k_responses} md_agreement_gen 
done
