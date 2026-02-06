#!/usr/bin/env python3
"""
Quick demo to showcase the etymology AI system using sample data.
""" 

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.phonetic_converter import PhoneticConverter


def main():
    print("=" * 70)
    print(" Etymology AI - Neural Phonetic Mapping Demo")
    print("=" * 70)
    
    # Load sample data
    sample_file = Path("data/raw/sample_etymology_data.json")
    if not sample_file.exists():
        print("✗ Sample data not found. Run: python3 src/data/sample_dataset.py")
        return
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n✓ Loaded {len(data)} cognate sets\n")
    
    # Initialize converter
    print("Initializing phonetic converter...")
    converter = PhoneticConverter()
    print("✓ Ready\n")
    
    # Show examples
    for idx, entry in enumerate(data[:3], 1):
        thai = entry['thai']
        sanskrit = entry['sanskrit']
        pie = entry['pie']
        
        print(f"\n[{idx}] {thai['meaning'].upper()}")
        print("-" * 70)
        
        # Thai
        thai_word = thai['word']
        thai_ipa = converter.to_ipa(thai_word, 'tha')
        print(f"  Thai:      {thai_word:15s} → IPA: {thai_ipa}")
        
        # Sanskrit
        san_word = sanskrit['word']
        san_ipa = converter.to_ipa(san_word, 'san')
        print(f"  Sanskrit:  {san_word:15s} → IPA: {san_ipa}")
        
        # PIE
        pie_word = pie['reconstructed']
        print(f"  PIE Root:  {pie_word}")
        
        # Modern cognates
        cognates = entry['cognates']
        for lang, info in cognates.items():
            word = info['word']
            ipa = info.get('ipa', converter.to_ipa(word, 'eng' if lang == 'english' else 'lat'))
            print(f"  {lang.capitalize():9s}  {word:15s} → IPA: {ipa}")
        
        # Calculate distances
        dist_thai_san = converter.phonetic_distance(thai_ipa, san_ipa)
        print(f"\n  📊 Phonetic Distance (Thai ↔ Sanskrit): {dist_thai_san:.2f}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Train phonetic embedding model on cognate pairs")
    print("  2. Build Siamese network for automatic cognate detection")
    print("  3. Create interactive etymology graph visualization")
    print("=" * 70)


if __name__ == "__main__":
    main()
