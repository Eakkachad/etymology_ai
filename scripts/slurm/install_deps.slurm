#!/bin/bash
#SBATCH --job-name=install_deps
#SBATCH --output=outputs/logs/install_%j.out
#SBATCH --error=outputs/logs/install_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --partition=defq

# One-time dependency installation in sax_nerf environment
echo "Installing Etymology AI dependencies in sax_nerf environment..."

cd /home/67070309/eak_project/etymology_ai

# Load miniconda3
module load miniconda3
eval "$(conda shell.bash hook)"
conda activate sax_nerf

# Install dependencies
pip install pytorch-lightning tensorboard pythainlp pyyaml

echo "Installation complete!"
python -c "import pytorch_lightning; print(f'PyTorch Lightning: {pytorch_lightning.__version__}')"
python -c "import pythainlp; print(f'PyThaiNLP: {pythainlp.__version__}')"
