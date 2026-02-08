#!/bin/bash
#SBATCH --job-name=etym_phonetic
#SBATCH --output=outputs/logs/phonetic_%j.out
#SBATCH --error=outputs/logs/phonetic_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=256G

# Etymology AI - Phonetic Embedding Training on DGX
# Author: Eakkachad
# Date: 2026-02-07

echo "=========================================="
echo "Starting Phonetic Embedding Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Setup environment
module load cuda/11.8  # Adjust based on DGX

cd /home/67070309/eak_project/etymology_ai

# Initialize conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

# Use existing sax_nerf environment (has PyTorch)
conda activate sax_nerf || { echo "Error: sax_nerf environment not found"; exit 1; }

# Install missing dependencies
echo "Installing Etymology AI dependencies..."
pip install pytorch-lightning tensorboard wandb pythainlp --quiet

# Print environment info
echo "Python: $(which python)"
echo "CUDA: $(nvcc --version | grep release)"
echo "GPUs: $(nvidia-smi --list-gpus | wc -l)"

# Create output directory
mkdir -p outputs/logs
mkdir -p outputs/phonetic_embedding/checkpoints

# Run training
python src/training/train_phonetic_embedding.py \
    --config configs/model_config.yaml \
    --data data/raw/sample_etymology_data.json \
    --output outputs/phonetic_embedding \
    --epochs 100 \
    --batch-size 256

echo "=========================================="
echo "Training completed!"
echo "=========================================="
