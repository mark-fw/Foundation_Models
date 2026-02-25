#!/usr/bin/zsh 

### Job Parameters 
#SBATCH --ntasks=1              
#SBATCH --time=00:15:00         
#SBATCH --job-name=prob_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=rwth0934  # Replace with your project-id or delete the line
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n23g0001
#SBATCH -p c23g

### Program Code
#---- activate conda
source ~/miniforge/etc/profile.d/conda.sh
conda activate torchgpu

#---- create log dir
mkdir -p logs

MODELFILE="/home/eu732103/master_thesis/hep_foundation_model/output/checkpoints/TOP_600000_best.pt"
INPUTFILE="/hpcwork/rwth0934/hep_foundation_model/preprocessed_data/Top_test_discrete_pT_eta_phi.h5"
OUTPUTFILE="output/plot_data/top_top.csv"

#print version of repo:
python util/gitversion.py

python compute_probabilities.py --model_path "$MODELFILE" --data_path "$INPUTFILE" --output_file "$OUTPUTFILE" --n_jets 50000 --batch_size 100 --input_key df