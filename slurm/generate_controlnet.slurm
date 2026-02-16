#!/bin/bash
#SBATCH -A med250011p
#SBATCH -J generate_controlnet
#SBATCH -p GPU-shared
#SBATCH --gres=gpu:1             # ✅ 아무 GPU 1개 요청 (권장 테스트용)
# #SBATCH --gres=gpu:h100-80:1   # 🔒 H100-80 GPU 요청 (대기 길 수 있음)
#SBATCH -t 12:00:00            # DDIM 포함 이미지 생성, 12시간 넉넉히
#SBATCH -o /ocean/projects/med230010p/mkwak/tumor-controller/logs/generate_controlnet_%j.out
#SBATCH -e /ocean/projects/med230010p/mkwak/tumor-controller/logs/generate_controlnet_%j.err

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

echo "[INFO] checking DDIM directory"
DDIM_DIR="/ocean/projects/med230010p/mkwak/normal-brain-recon/image_generation/2025-09-27_17-43-02/ddim"
if [ -d "$DDIM_DIR" ]; then
    echo "[INFO] DDIM directory found: $DDIM_DIR"
    DDIM_COUNT=$(ls -1 "$DDIM_DIR"/*.npy 2>/dev/null | wc -l)
    echo "[INFO] Found $DDIM_COUNT DDIM files"
else
    echo "[WARNING] DDIM directory not found: $DDIM_DIR"
    echo "[WARNING] DDIM images will not be available in overview plots"
fi

echo "[INFO] starting ControlNet image generation with DDIM integration"
python generate_controlnet.py \
  --controlnet-id "2025-08-08_18-52-26" \
  --num-steps 50 \
  --strength 1.0 \
  --guidance-scale 7.5 \
  --controlnet-conditioning-scale 0.8 \
  --expanded-sigma 5.0 5.0 \
  --expanded-threshold 0.05 \
  --ddim-image-dir "$DDIM_DIR"
# python generate_controlnet.py \
#   --controlnet-id "2025-08-08_18-52-26" \
#   --num-steps 40 \
#   --strength 1.0 \
#   --guidance-scale 7.5 \
#   --controlnet-conditioning-scale 0.8 \
#   --expanded-sigma 3.0 3.0 \
#   --expanded-threshold 0.05 \
#   --ddim-image-dir "$DDIM_DIR"
if [ $? -ne 0 ]; then
    echo "[ERROR] Python execution failed"; exit 1
fi

echo "[INFO] ControlNet generation with DDIM integration completed successfully"
echo "[INFO] Check the image_generation directory for results with DDIM visualizations"
