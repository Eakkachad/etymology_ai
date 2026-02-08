#!/bin/bash
#SBATCH --job-name=etym_siamese
#SBATCH --output=outputs/logs/siamese_%j.out
#SBATCH --error=outputs/logs/siamese_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=256G

echo "=========================================="
echo "Starting Siamese Network Training"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Setup
module load cuda/11.8
cd /home/67070309/eak_project/etymology_ai

# Initialize and activate conda
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate sax_nerf

mkdir -p outputs/logs
mkdir -p outputs/siamese/checkpoints

# Run training
python src/training/train_siamese.py \
    --config configs/model_config.yaml \
    --data data/raw/sample_etymology_data.json \
    --encoder outputs/phonetic_embedding/checkpoints/best.ckpt \
    --output outputs/siamese

echo "Training completed!"
