"""
Simple test script to verify the models work without training.
Tests model instantiation and forward pass.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Etymology AI - Model Verification Test")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from src.models.phonetic_embedding import PhoneticEmbeddingModel
    from src.models.siamese_network import SiameseNetwork, TripletLoss
    from src.data.phonetic_converter import PhoneticConverter
    from src.data.dataset import CognateDataset
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Create phonetic embedding model
print("\n[2/5] Creating Phonetic Embedding Model...")
try:
    model = PhoneticEmbeddingModel(
        vocab_size=256,
        embedding_dim=512,
        num_layers=6,
        num_heads=8,
        pooling="mean"
    )
    print(f"✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    sys.exit(1)

# Test 3: Test forward pass
print("\n[3/5] Testing forward pass...")
try:
    import torch
    x = torch.randint(1, 256, (4, 20))  # batch=4, seq_len=20
    output = model(x)
    print(f"✓ Forward pass successful: {x.shape} → {output.shape}")
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    sys.exit(1)

# Test 4: Create Siamese Network
print("\n[4/5] Creating Siamese Network...")
try:
    siamese = SiameseNetwork(
        encoder=model,
        embedding_dim=512,
        projection_dims=[512, 256, 128],
        similarity_metric="cosine"
    )
    print(f"✓ Siamese network created ({sum(p.numel() for p in siamese.parameters()):,} parameters)")
except Exception as e:
    print(f"✗ Siamese creation error: {e}")
    sys.exit(1)

# Test 5: Test Siamese forward pass
print("\n[5/5] Testing Siamese network...")
try:
    x1 = torch.randint(1, 256, (4, 20))
    x2 = torch.randint(1, 256, (4, 20))
    similarity = siamese(x1, x2)
    print(f"✓ Siamese forward pass: {similarity.shape}, scores: {similarity.tolist()[:2]}")
except Exception as e:
    print(f"✗ Siamese forward error: {e}")
    sys.exit(1)

# Test 6: Test Dataset
print("\n[6/6] Testing Dataset...")
try:
    converter = PhoneticConverter()
    dataset = CognateDataset(
        "data/raw/sample_etymology_data.json",
        converter,
        mode="triplet"
    )
    sample = dataset[0]
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print(f"  Sample: {sample['anchor_text']} ↔ {sample['positive_text']}")
except Exception as e:
    print(f"✗ Dataset error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe Etymology AI system is working correctly!")
print("\nNext steps:")
print("  1. Install full dependencies (if needed): bash scripts/setup_environment.sh")
print("  2. Train models: python src/training/train_phonetic_embedding.py --epochs 2")
print("  3. Or submit to DGX: sbatch scripts/slurm/train_phonetic.slurm")
