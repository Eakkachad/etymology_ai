#!/usr/bin/env python3
"""
Quick demo script to download and explore sample etymology data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.downloaders import KaikkiDownloader
import json


def main():
    print("=" * 60)
    print("Etymology AI - Sample Data Download Demo")
    print("=" * 60)
    
    # Download Thai data
    print("\n[1/3] Downloading Thai etymology data from Kaikki...")
    kaikki = KaikkiDownloader()
    thai_entries = kaikki.load_etymology_data("Thai")
    
    print(f"\n✓ Found {len(thai_entries)} Thai words with etymology information")
    
    # Show examples
    print("\n[2/3] Sample Thai words with Pali/Sanskrit origins:")
    print("-" * 60)
    
    count = 0
    for entry in thai_entries:
        # Look for Pali or Sanskrit in etymology
        etym_text = entry.get('etymology_text', '').lower()
        
        if 'pali' in etym_text or 'sanskrit' in etym_text:
            word = entry.get('word', 'N/A')
            pos = entry.get('pos', 'N/A')
            
            print(f"\nWord: {word} ({pos})")
            print(f"Etymology: {entry.get('etymology_text', 'N/A')[:200]}...")
            
            count += 1
            if count >= 5:
                break
    
    # Download Sanskrit for comparison
    print("\n\n[3/3] Downloading Sanskrit data for comparison...")
    sanskrit_entries = kaikki.load_etymology_data("Sanskrit")
    print(f"✓ Found {len(sanskrit_entries)} Sanskrit entries")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Thai entries: {len(thai_entries)}")
    print(f"  Sanskrit entries: {len(sanskrit_entries)}")
    print(f"  Data location: data/raw/")
    print("=" * 60)
    
    print("\n✓ Setup complete! Next step: Convert to IPA")
    print("  Run: jupyter notebook notebooks/01_phonetic_exploration.ipynb")


if __name__ == "__main__":
    main()
