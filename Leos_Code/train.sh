#!/bin/bash
#SBATCH --job-name=Top_Train_Val
#SBATCH --output=logs/Top_training.txt
#SBATCH --partition=c23g           
#SBATCH --exclude=n23g0001
#SBATCH --gres=gpu:1               # Anzahl GPUs
#SBATCH --cpus-per-task=4         # Threads für DataLoader
##SBATCH --mem=120G                  
#SBATCH --time=12:00:00            
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Mfanselow2001@web.de

### Program Code

#module purge
#module load CUDA/12.6.3

source /home/ew640340/miniforge3/etc/profile.d/conda.sh
conda activate gpu-env-new

#export CPATH=$CUDA_HOME/include:$CPATH
#export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH

#export OMP_NUM_THREADS=1
#export MKL_NUM_THREADS=$OMP_NUM_THREADS
#export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
#export NCCL_DEBUG=warning
#export PYTHONFAULTHANDLER=1

# nvidia-smi

Data="Top"

INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/${Data}_train_discrete_pT_eta_phi.h5"
OUTPUTPATH="/hpcwork/thes1906/Ph.D./transformer_output"
NAME="${Data}_600k_train_200k_val"

python train.py --data_path "$INPUTFILE" --output_path "$OUTPUTPATH" --name "$NAME" --num_epochs 50 --batch_size 100 --num_jets_train 600000 --num_jets_val 200000
# python train_new.py --data_path "$INPUTFILE" --output_path "$OUTPUTPATH" --name "$NAME" --num_epochs 50 --batch_size 100 

# python train.py --num_epochs 50 --num_const 50 --num_jets_train 100 --num_jets_val 100 --batch_size 20 --name "QCD"








