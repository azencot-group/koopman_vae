#!/bin/bash

###############################################################################################	
########
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
###############################################################################################

#SBATCH --partition rtx3090			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --qos azencot
#SBATCH --time 7-00:00:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name hyperparameters_optimization	### name of the job
#SBATCH --output train-%J.out			### output log for running job - %J for job number
#SBATCH --mail-user=gdolim@post.bgu.ac.il    ### User's email for sending job status
#SBATCH --mail-type=FAIL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gpus=rtx_3090:3			# 3 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --mem=60G				### amount of RAM memory, allocating more than 60G requires IT team's permission
##SBATCH --cpus-per-task=4			### number of CPU cores, allocating more than 10G requires IT team's permission

##SBATCH --exclude=dt-1080-01,ise-1080-04,ise-1080-01,ise-1080-05,ise-1080-03,ise-1080-02,cs-1080-04,cs-1080-05,cs-1080-01,cs-1080-02,cs-1080-03,dt-gpu-02,dt-gpu-07,cs-gpu-01

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your src_code below ####
module load anaconda		### load anaconda module (must be present when working with conda environments)
source activate /home/gdolim/.conda/envs/koopman_vae_3_9	### activate a conda environment, replace my_env with your conda environment

# Set the CUDA devices visible to the job
export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | tr ',' ' ')
unset LOCAL_RANK		# Sometimes this emvironment variable causes lightning not to detect GPUs.
export NCCL_P2P_DISABLE=1	# This solves the problem in lightning on 4090.

srun python /cs/cs_groups/azencot_group/inon/koopman_vae/optimize_hyperparams.py --epochs 4 --lr 0.001 --batch_size 512 --evl_interval 2 --n_trials 3 --no-prior-sampling

