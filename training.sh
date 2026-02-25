#!/bin/bash
#SBATCH --job-name=Device2
##SBATCH --output=Train_Val.txt
#SBATCH --output=Device2.txt
#SBATCH --partition=c23g           
#SBATCH --exclude=n23g0001
#SBATCH --gres=gpu:1               # Anzahl GPUs
#SBATCH --cpus-per-task=1         # Threads für DataLoader
##SBATCH --mem=120G                  
#SBATCH --time=00:02:00            
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=Mfanselow2001@web.de

### Program Code

source /home/ew640340/miniforge3/etc/profile.d/conda.sh
# conda activate Test1
conda activate gpu-env


#export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
#export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK


#python '/home/ew640340/Ph.D./Foundation_Models/Train_Val.py'

python '/home/ew640340/Ph.D./Foundation_Models/t.py'

### cd ~
### cd ..
### cd 'Ph.D./Foundation_Models'
### sbatch training.sh
### sacct
### scancel Job nr
### scancel --me (alle)
### sacct -X -j JobID -o AllocTRES%100