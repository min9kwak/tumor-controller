#!/bin/bash
#SBATCH -A med250011p
#SBATCH -J preprocess_t1
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1
#SBATCH -t 3:00:00
#SBATCH -o /ocean/projects/med230010p/mkwak/tumor-controller/logs/preprocess_t1_%j.out
#SBATCH -e /ocean/projects/med230010p/mkwak/tumor-controller/logs/preprocess_t1_%j.err

mkdir -p /ocean/projects/med230010p/mkwak/tumor-controller/logs

module purge
module load anaconda3

cd /ocean/projects/med230010p/mkwak/tumor-controller

source /ocean/projects/med230010p/mkwak/miniconda3/etc/profile.d/conda.sh
conda activate tumor

echo "[INFO] Starting T1 preprocessing"
python dataset/preprocessing/1_prepare_image.py

echo "[INFO] Preprocessing completed"
