#!/bin/bash -l
# NOTE the -l flag!
# This is an example job file for a single core CPU bound program
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler
# Please copy this file to your home directory and modify it 
# to suit your needs.
#
# If you need any help, please [submit a ticket](https://help.rit.edu/sp?id=rc_request) or contact us on Slack.
#
# Name of the job - You'll probably want to customize this.
#SBATCH -J sample-$1-$2-$3
# Standard out and Standard Error output files
#SBATCH -o test.o
#SBATCH -e test.e
# To send slack notifications, set the address below and remove one of the '#' sings
##SBATCH --mail-user=slack:@cmhvcs
# notify on state change: BEGIN, END, FAIL, OR ALL
# 5 days is the run time MAX, anything over will be KILLED unless you talk with RC
# Request 4 days and 5 hours
#SBATCH -t 4-5:0:0
#Put the job in the appropriate partition matching the account and request one core
#SBATCH -A population -p tier3 -c 1
#Job membory requirements in MB=m (default), GB=g, or TB=t
#SBATCH --mem=48g
#
# Your job script goes below this line.
#
conda activate vet
python3 /home/cmhvcs/vet/response_resampler.py --config_file=acl2024_config.csv --n_items=$1 --k_responses=$2 --exp_dir= --input_response_file=responses_simulated_distr_dist=$3_gen_N=$1_K=$2_num_samples=1000.json 

