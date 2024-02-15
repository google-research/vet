#!/bin/bash -l
for d in 0.005 0.01 0.02 0.1
do
        sbatch ./resample.sh 100 10 ${d}         
        sbatch ./resample.sh 1000 1 ${d}         
        sbatch ./resample.sh 25 100 ${d}         
        sbatch ./resample.sh 100 25 ${d}         
        sbatch ./resample.sh 500 5 ${d}         
        sbatch ./resample.sh 50 100 ${d}         
        sbatch ./resample.sh 1000 5 ${d}         
        sbatch ./resample.sh 100 100 ${d}         
        sbatch ./resample.sh 1000 10 ${d}         
        sbatch ./resample.sh 250 100 ${d}         
        sbatch ./resample.sh 1000 25 ${d}         
        sbatch ./resample.sh 500 100 ${d}         
        sbatch ./resample.sh 1000 50 ${d}         
done
    
