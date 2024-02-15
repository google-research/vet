#!/bin/bash -l
for d in 0.005 0.01 0.02 0.1
do
        sbatch ./sample.sh 100 10 ${d}         
        sbatch ./sample.sh 1000 1 ${d}         
        sbatch ./sample.sh 25 100 ${d}         
        sbatch ./sample.sh 100 25 ${d}         
        sbatch ./sample.sh 500 5 ${d}         
        sbatch ./sample.sh 50 100 ${d}         
        sbatch ./sample.sh 1000 5 ${d}         
        sbatch ./sample.sh 100 100 ${d}         
        sbatch ./sample.sh 1000 10 ${d}         
        sbatch ./sample.sh 250 100 ${d}         
        sbatch ./sample.sh 1000 25 ${d}         
        sbatch ./sample.sh 500 100 ${d}         
        sbatch ./sample.sh 1000 50 ${d}         
done
    
