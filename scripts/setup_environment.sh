#!/bin/bash
# Setup script for Etymology AI on DGX
# Run this once to install all dependencies

set -e

echo "=========================================="
echo "Etymology AI - Environment Setup"
echo "=========================================="

cd /home/67070309/eak_project/etymology_ai

# Create Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate environment
source venv/bin/activate

echo "Virtual environment created at: $(pwd)/venv"
echo "Python: $(which python)"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for quick install, GPU version if CUDA available)
echo ""
echo "Installing PyTorch..."
if command -v nvcc &> /dev/null; then
    # CUDA available
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "CUDA ${CUDA_VERSION} detected, installing GPU version..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    # CPU only
    echo "No CUDA detected, installing CPU version..."
    pip install torch torchvision
fi

# Install PyTorch Lightning
echo ""
echo "Installing PyTorch Lightning..."
pip install pytorch-lightning tensorboard wandb

# Install torch-geometric (may take a while)
echo ""
echo "Installing torch-geometric..."
pip install torch-geometric torch-scatter torch-sparse || echo "Warning: torch-geometric installation skipped"

# Install NLP tools
echo ""
echo "Installing NLP libraries..."
pip install pythainlp epitran panphon

# Install data processing
echo ""
echo "Installing data processing libraries..."
pip install pandas numpy pyyaml tqdm requests beautifulsoup4 lxml

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch_lightning; print(f'PyTorch Lightning: {pytorch_lightning.__version__}')"
python -c "import pythainlp; print(f'PyThaiNLP: {pythainlp.__version__}')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To use this environment:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run training:"
echo "  python src/training/train_phonetic_embedding.py --epochs 2 --batch-size 4"
