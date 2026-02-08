#!/usr/bin/env python3
"""
Demo script showing Etymology AI system capabilities:
1. Phonetic conversion (Thai/Sanskrit -> IPA)
2. Model forward pass (embeddings)
3. Cognate similarity scoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from src.data.phonetic_converter import PhoneticConverter
from src.models.phonetic_embedding import PhoneticEmbeddingModel
from src.models.siamese_network import SiameseNetwork

def demo_phonetic_conversion():
    """Demo: Convert words to IPA"""
    print("=" * 60)
    print("DEMO 1: Phonetic Conversion (Word -> IPA)")
    print("=" * 60)
    
    converter = PhoneticConverter()
    
    # Test words
    test_pairs = [
        ("tri", "san", "three", "eng"),
        ("mātr̥", "san", "mother", "eng"),
        ("pitar", "san", "father", "eng"),
    ]
    
    for w1, lang1, w2, lang2 in test_pairs:
        ipa1 = converter.to_ipa(w1, lang1)
        ipa2 = converter.to_ipa(w2, lang2)
        distance = converter.phonetic_distance(ipa1, ipa2)
        
        print(f"\n{w1} ({lang1}) -> IPA: {ipa1}")
        print(f"{w2} ({lang2}) -> IPA: {ipa2}")
        print(f"Phonetic distance: {distance:.3f}")
    
    return converter

def demo_model_embeddings(converter):
    """Demo: Generate phonetic embeddings"""
    print("\n" + "=" * 60)
    print("DEMO 2: Phonetic Embeddings (IPA -> Vector)")
    print("=" * 60)
    
    # Create model
    model = PhoneticEmbeddingModel(
        vocab_size=256,
        embedding_dim=128,
        num_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_length=50
    )
    model.eval()
    
    # Convert word to IPA then to tensor
    word = "tri"
    ipa = converter.to_ipa(word, "san")
    
    # Simple encoding: convert IPA chars to integers
    encoded = torch.tensor([[ord(c) % 256 for c in ipa[:10]]], dtype=torch.long)
    
    # Get embedding
    with torch.no_grad():
        embedding = model(encoded)
    
    print(f"\nWord: {word}")
    print(f"IPA: {ipa}")
    print(f"Encoded: {encoded[0].tolist()}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 10 dims): {embedding[0, :10].tolist()}")
    
    return model

def demo_cognate_detection(model):
    """Demo: Detect cognates using Siamese network"""
    print("\n" + "=" * 60)
    print("DEMO 3: Cognate Detection (Similarity Scoring)")
    print("=" * 60)
    
    # Create Siamese network
    siamese = SiameseNetwork(
        encoder=model,
        projection_dims=[128, 64],
        similarity_metric="cosine"
    )
    siamese.eval()
    
    # Test pairs (cognates vs non-cognates)
    test_data = [
        {
            "pair": ("tri", "three"),
            "is_cognate": True,
            "note": "Sanskrit/English cognate (PIE *tréyes)"
        },
        {
            "pair": ("mātr̥", "mother"),
            "is_cognate": True,
            "note": "Sanskrit/English cognate (PIE *méh₂tēr)"
        },
        {
            "pair": ("tri", "mother"),
            "is_cognate": False,
            "note": "Not cognates"
        }
    ]
    
    converter = PhoneticConverter()
    
    for item in test_data:
        w1, w2 = item["pair"]
        
        # Simple encoding
        ipa1 = converter.to_ipa(w1, "san")
        ipa2 = converter.to_ipa(w2, "eng")
        
        enc1 = torch.tensor([[ord(c) % 256 for c in ipa1[:10]]], dtype=torch.long)
        enc2 = torch.tensor([[ord(c) % 256 for c in ipa2[:10]]], dtype=torch.long)
        
        # Get similarity using forward method
        with torch.no_grad():
            similarity = siamese(enc1, enc2)
        
        print(f"\n{w1} <-> {w2}")
        print(f"  IPA: {ipa1} <-> {ipa2}")
        print(f"  Expected: {'COGNATE' if item['is_cognate'] else 'NOT COGNATE'}")
        print(f"  Similarity: {similarity.item():.4f}")
        print(f"  Note: {item['note']}")

def demo_data_loading():
    """Demo: Load and process sample data"""
    print("\n" + "=" * 60)
    print("DEMO 4: Data Loading & Processing")
    print("=" * 60)
    
    data_path = Path(__file__).parent.parent / "data/raw/sample_etymology_data.json"
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    with open(data_path) as f:
        data = json.load(f)
    
    print(f"\nLoaded {len(data)} cognate pairs:")
    
    for i, entry in enumerate(data[:3], 1):
        w1 = entry['word1']
        w2 = entry['word2']
        print(f"\n{i}. {w1['text']} ({w1['language']}) <-> {w2['text']} ({w2['language']})")
        print(f"   IPA: {w1['ipa']} <-> {w2['ipa']}")
        print(f"   Etymology: {entry['etymology']}")

def main():
    print("\n" + "=" * 60)
    print("Etymology AI - System Demo")
    print("=" * 60)
    
    try:
        # Run demos
        converter = demo_phonetic_conversion()
        model = demo_model_embeddings(converter)
        demo_cognate_detection(model)
        demo_data_loading()
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        print("\nNote: Models shown here use random weights (not trained).")
        print("For actual cognate detection, models need to be trained on real data.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
