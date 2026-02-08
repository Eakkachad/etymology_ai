#!/bin/bash
#SBATCH --job-name=etym_train
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=defq

echo "=========================================="
echo "Etymology AI - Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

cd /home/67070309/eak_project/etymology_ai

# Load environment (NO pip install - dependencies already installed!)
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate sax_nerf

# Print environment
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "PyTorch Lightning: $(python -c 'import pytorch_lightning; print(pytorch_lightning.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Create output directories
mkdir -p outputs/logs outputs/test_run/checkpoints

# Run training (2 epochs for quick test)
echo "Starting training..."
echo "=========================================="
python src/training/train_phonetic_embedding.py \
    --config configs/model_config.yaml \
    --data data/raw/sample_etymology_data.json \
    --output outputs/test_run \
    --epochs 2 \
    --batch-size 4 \
    --devices 1

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed!"
    ls -lh outputs/test_run/checkpoints/
else
    echo "✗ Training failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
