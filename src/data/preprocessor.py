"""
Data preprocessing utilities for Etymology AI.

Processes downloaded data into training-ready formats:
- Extract cognate pairs from Kaikki etymology data
- Build graph structures from etymological relationships
- Cache processed datasets
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EtymologyPreprocessor:
    """
    Preprocess raw etymology data into structured formats.
    """
    
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def process_kaikki_data(
        self,
        input_file: str,
        output_file: str,
        language: str = "thai"
    ) -> List[Dict]:
        """
        Extract etymology information from Kaikki JSONL file.
        
        Args:
            input_file: Path to Kaikki JSONL file
            output_file: Path to save processed JSON
            language: Source language
        
        Returns:
            List of etymology entries with structured data
        """
        entries = []
        
        logger.info(f"Processing Kaikki {language} data from {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Parsing {language}"):
                try:
                    entry = json.loads(line.strip())
                    
                    # Filter entries with etymology information
                    if 'etymology_text' not in entry and 'etymology_templates' not in entry:
                        continue
                    
                    # Extract relevant fields
                    processed = {
                        'word': entry.get('word', ''),
                        'pos': entry.get('pos', ''),
                        'language': language,
                        'etymology_text': entry.get('etymology_text', ''),
                        'etymology_templates': entry.get('etymology_templates', []),
                        'senses': entry.get('senses', [])
                    }
                    
                    # Try to extract source language info
                    source_langs = self._extract_source_languages(entry)
                    if source_langs:
                        processed['source_languages'] = source_langs
                    
                    entries.append(processed)
                
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Extracted {len(entries)} entries with etymology")
        
        # Save to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved processed data to {output_path}")
        return entries
    
    def _extract_source_languages(self, entry: Dict) -> List[str]:
        """Extract source language codes from etymology templates."""
        source_langs = set()
        
        templates = entry.get('etymology_templates', [])
        for template in templates:
            if 'args' in template:
                args = template['args']
                # Common template structures: {"1": "lang_code", ...}
                if '1' in args:
                    source_langs.add(args['1'])
        
        return list(source_langs)
    
    def build_cognate_pairs(
        self,
        etymology_data: List[Dict],
        min_pair_confidence: float = 0.7
    ) -> List[Tuple[Dict, Dict, float]]:
        """
        Build cognate pairs from etymology data.
        
        Args:
            etymology_data: List of etymology entries
            min_pair_confidence: Minimum confidence score to include pair
        
        Returns:
            List of (word1, word2, confidence) tuples
        """
        pairs = []
        
        # Group words by shared etymology
        etymology_groups = defaultdict(list)
        
        for entry in etymology_data:
            # Use etymology text as grouping key (simplified)
            etym_key = entry.get('etymology_text', '')[:100]  # First 100 chars
            if etym_key:
                etymology_groups[etym_key].append(entry)
        
        # Create pairs within each group
        for group_words in etymology_groups.values():
            if len(group_words) < 2:
                continue
            
            # Create all pairwise combinations
            for i in range(len(group_words)):
                for j in range(i + 1, len(group_words)):
                    word1 = group_words[i]
                    word2 = group_words[j]
                    
                    # Assign confidence (could be refined with NLP)
                    confidence = 0.8  # Placeholder
                    
                    if confidence >= min_pair_confidence:
                        pairs.append((word1, word2, confidence))
        
        logger.info(f"Built {len(pairs)} cognate pairs")
        return pairs
    
    def build_etymology_graph(
        self,
        etymology_data: List[Dict],
        output_file: str
    ) -> Dict:
        """
        Build graph structure from etymology data.
        
        Returns:
            Dict with 'nodes' and 'edges' lists
        """
        nodes = []
        edges = []
        word_to_idx = {}
        
        # Add all words as nodes
        for entry in etymology_data:
            word = entry['word']
            if word not in word_to_idx:
                idx = len(nodes)
                word_to_idx[word] = idx
                nodes.append({
                    'id': idx,
                    'word': word,
                    'language': entry['language'],
                    'pos': entry.get('pos', ''),
                    'etymology': entry.get('etymology_text', '')
                })
        
        # Add edges based on etymology relationships
        # Simplified: Connect words with shared etymology text
        etymology_groups = defaultdict(list)
        for entry in etymology_data:
            etym_key = entry.get('etymology_text', '')[:100]
            if etym_key:
                etymology_groups[etym_key].append(entry['word'])
        
        for group_words in etymology_groups.values():
            if len(group_words) < 2:
                continue
            
            # Fully connect words in same etymology group
            for i in range(len(group_words)):
                for j in range(i + 1, len(group_words)):
                    word1 = group_words[i]
                    word2 = group_words[j]
                    
                    if word1 in word_to_idx and word2 in word_to_idx:
                        idx1 = word_to_idx[word1]
                        idx2 = word_to_idx[word2]
                        
                        edges.append({
                            'source': idx1,
                            'target': idx2,
                            'type': 'cognate'
                        })
        
        graph = {
            'nodes': nodes,
            'edges': edges
        }
        
        # Save graph
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Built graph: {len(nodes)} nodes, {len(edges)} edges")
        logger.info(f"Saved graph to {output_path}")
        
        return graph
    
    def cache_processed_data(
        self,
        data: any,
        cache_name: str
    ) -> Path:
        """
        Cache processed data using pickle.
        
        Args:
            data: Data to cache
            cache_name: Name for cache file (without extension)
        
        Returns:
            Path to cached file
        """
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Cached data to {cache_file}")
        return cache_file
    
    def load_cached_data(self, cache_name: str) -> any:
        """Load cached data."""
        cache_file = self.cache_dir / f"{cache_name}.pkl"
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"Loaded cached data from {cache_file}")
        return data


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = EtymologyPreprocessor()
    
    # Check if Kaikki data exists
    kaikki_file = Path("data/raw/kaikki_thai.jsonl")
    
    if kaikki_file.exists():
        print("Processing Kaikki Thai data...")
        entries = preprocessor.process_kaikki_data(
            input_file=str(kaikki_file),
            output_file="data/processed/thai_etymology.json",
            language="thai"
        )
        
        print("\nBuilding cognate pairs...")
        pairs = preprocessor.build_cognate_pairs(entries)
        print(f"Created {len(pairs)} pairs")
        
        print("\nBuilding etymology graph...")
        graph = preprocessor.build_etymology_graph(
            entries,
            output_file="data/processed/etymology_graph.json"
        )
        
    else:
        print(f"Kaikki data not found at {kaikki_file}")
        print("Please download data first using:")
        print("  python scripts/download_sample_data.py --source kaikki --language thai")
