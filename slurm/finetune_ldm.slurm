#!/bin/bash
#SBATCH -A med230010p
#SBATCH -J finetune_ldm
#SBATCH -p GPU-shared
# #SBATCH --gres=gpu:1             # ✅ 아무 GPU 1개 요청 (권장 테스트용)
#SBATCH --gres=gpu:h100-80:1   # 🔒 H100-80 GPU 요청 (대기 길 수 있음)
#SBATCH -t 8:00:00
#SBATCH -o /ocean/projects/med230010p/mkwak/tumor-controller/logs/finetune_ldm_%j.out
#SBATCH -e /ocean/projects/med230010p/mkwak/tumor-controller/logs/finetune_ldm_%j.err

# 로그 디렉토리 미리 생성
mkdir -p /ocean/projects/med230010p/mkwak/tumor-controller/logs

module purge
module load anaconda3
module load cuda

cd "$SLURM_SUBMIT_DIR"

echo "[INFO] loading conda"
source /ocean/projects/med230010p/mkwak/miniconda3/etc/profile.d/conda.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] conda source failed"; exit 1
fi

echo "[INFO] activating tumor environment"
conda activate tumor
if [ $? -ne 0 ]; then
    echo "[ERROR] conda activate failed"; exit 1
fi

echo "[INFO] printing conda environment"
echo "CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"

echo "[INFO] printing GPU info"
nvidia-smi

echo "[INFO] starting Python training"
python finetune_ldm.py
if [ $? -ne 0 ]; then
    echo "[ERROR] Python execution failed"; exit 1
fi

echo "[INFO] job completed successfully"
