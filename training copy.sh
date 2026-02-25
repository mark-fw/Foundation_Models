#!/bin/bash

### Job Parameters 

###SBATCH -A rwth0934
#SBATCH --job-name=test
#SBATCH --output=test.txt
#SBATCH --time=00:01:00
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Mfanselow2001@web.de
###SBATCH --mem=16G

### Program Code

source /home/ew640340/miniforge3/etc/profile.d/conda.sh
conda activate Test1

python 't.py'

### cd ~
### cd ..
### cd 'Ph.D./Foundation_Models'
### sbatch 'training copy.sh'
### sacct
### scancel Job nr
### scancel --me (alle)