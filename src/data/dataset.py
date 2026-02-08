"""
PyTorch Dataset classes for Etymology AI training.

This module provides:
- CognateDataset: For training siamese networks with cognate pairs
- EtymologyGraphDataset: For training GNN on etymology graphs
"""

import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)


class CognateDataset(Dataset):
    """
    Dataset for cognate detection training.
    
    Provides word pairs/triplets with labels:
    - Positive pairs: Known cognates from same etymology
    - Negative pairs: Random non-cognate words
    - Triplets: (anchor, positive, negative) for triplet loss
    """
    
    def __init__(
        self,
        data_path: str,
        phonetic_converter,
        mode: str = "triplet",
        negative_ratio: int = 2,
        max_phoneme_length: int = 50
    ):
        """
        Args:
            data_path: Path to etymology JSON data
            phonetic_converter: PhoneticConverter instance
            mode: "triplet" or "pair"
            negative_ratio: Number of negative samples per positive
            max_phoneme_length: Maximum IPA sequence length
        """
        self.phonetic_converter = phonetic_converter
        self.mode = mode
        self.negative_ratio = negative_ratio
        self.max_phoneme_length = max_phoneme_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # Build cognate pairs and word list
        self.cognate_pairs = []
        self.all_words = []
        self._build_pairs()
        
        logger.info(f"Loaded {len(self.cognate_pairs)} cognate pairs")
        logger.info(f"Total vocabulary: {len(self.all_words)} words")
    
    def _build_pairs(self):
        """Extract cognate pairs from etymology data."""
        for entry in self.raw_data:
            # Extract all words in this cognate set
            words = []
            
            # Thai word
            if 'thai' in entry:
                words.append({
                    'word': entry['thai']['word'],
                    'lang': 'tha',
                    'meaning': entry['thai'].get('meaning', '')
                })
            
            # Sanskrit word
            if 'sanskrit' in entry:
                words.append({
                    'word': entry['sanskrit']['word'],
                    'lang': 'san',
                    'meaning': entry.get('thai', {}).get('meaning', '')
                })
            
            # Modern cognates
            if 'cognates' in entry:
                for lang, info in entry['cognates'].items():
                    lang_code = 'eng' if lang == 'english' else 'lat' if lang == 'latin' else 'grc'
                    words.append({
                        'word': info['word'],
                        'lang': lang_code,
                        'meaning': entry.get('thai', {}).get('meaning', '')
                    })
            
            # Create all pairwise combinations as positive pairs
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    self.cognate_pairs.append((words[i], words[j], 1.0))  # 1.0 = positive
            
            # Add to vocabulary
            self.all_words.extend(words)
    
    def _convert_to_ipa(self, word_dict: Dict) -> str:
        """Convert word to IPA representation."""
        word = word_dict['word']
        lang = word_dict['lang']
        
        try:
            ipa = self.phonetic_converter.to_ipa(word, lang)
            return ipa
        except:
            return word  # Fallback to original
    
    def _encode_ipa(self, ipa: str) -> torch.Tensor:
        """
        Encode IPA string to character indices.
        
        Returns:
            Tensor of shape (max_phoneme_length,) with character codes
        """
        # Simple character-level encoding
        chars = list(ipa)[:self.max_phoneme_length]
        
        # Convert to ASCII codes (simplified, could use IPA-specific vocab)
        codes = [ord(c) for c in chars]
        
        # Pad to max length
        while len(codes) < self.max_phoneme_length:
            codes.append(0)  # 0 = padding
        
        return torch.tensor(codes, dtype=torch.long)
    
    def __len__(self) -> int:
        if self.mode == "triplet":
            return len(self.cognate_pairs)
        else:
            return len(self.cognate_pairs) * (1 + self.negative_ratio)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "triplet":
            return self._get_triplet(idx)
        else:
            return self._get_pair(idx)
    
    def _get_triplet(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get triplet: (anchor, positive, negative)
        """
        # Get positive pair
        anchor_word, positive_word, _ = self.cognate_pairs[idx]
        
        # Sample random negative (not a cognate)
        negative_word = random.choice(self.all_words)
        
        # Convert to IPA and encode
        anchor_ipa = self._convert_to_ipa(anchor_word)
        positive_ipa = self._convert_to_ipa(positive_word)
        negative_ipa = self._convert_to_ipa(negative_word)
        
        return {
            'anchor': self._encode_ipa(anchor_ipa),
            'positive': self._encode_ipa(positive_ipa),
            'negative': self._encode_ipa(negative_ipa),
            'anchor_text': anchor_word['word'],
            'positive_text': positive_word['word'],
            'negative_text': negative_word['word']
        }
    
    def _get_pair(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get pair: (word1, word2, label)
        """
        pair_idx = idx // (1 + self.negative_ratio)
        is_positive = (idx % (1 + self.negative_ratio)) == 0
        
        if is_positive:
            word1, word2, label = self.cognate_pairs[pair_idx]
        else:
            # Negative pair
            word1 = self.cognate_pairs[pair_idx][0]
            word2 = random.choice(self.all_words)
            label = 0.0
        
        ipa1 = self._convert_to_ipa(word1)
        ipa2 = self._convert_to_ipa(word2)
        
        return {
            'word1': self._encode_ipa(ipa1),
            'word2': self._encode_ipa(ipa2),
            'label': torch.tensor(label, dtype=torch.float32),
            'word1_text': word1['word'],
            'word2_text': word2['word']
        }


class EtymologyGraphDataset(Dataset):
    """
    Dataset for GNN training on etymology graphs.
    
    Provides graph structure with:
    - Nodes: Words across all languages
    - Edges: Known etymological relationships
    - Features: Phonetic embeddings
    """
    
    def __init__(
        self,
        data_path: str,
        phonetic_converter,
        embedding_dim: int = 512
    ):
        """
        Args:
            data_path: Path to etymology JSON data
            phonetic_converter: PhoneticConverter instance
            embedding_dim: Dimension of node features
        """
        self.phonetic_converter = phonetic_converter
        self.embedding_dim = embedding_dim
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # Build graph
        self.nodes = []  # List of word dicts
        self.edges = []  # List of (src_idx, dst_idx) tuples
        self._build_graph()
        
        logger.info(f"Built graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _build_graph(self):
        """Build graph structure from etymology data."""
        node_index = {}  # word -> node_idx mapping
        
        for entry in self.raw_data:
            entry_nodes = []
            
            # Add Thai node
            if 'thai' in entry:
                thai_word = entry['thai']['word']
                if thai_word not in node_index:
                    idx = len(self.nodes)
                    node_index[thai_word] = idx
                    self.nodes.append({
                        'word': thai_word,
                        'lang': 'tha',
                        'lang_family': 'tai_kadai'
                    })
                entry_nodes.append(node_index[thai_word])
            
            # Add Sanskrit node
            if 'sanskrit' in entry:
                san_word = entry['sanskrit']['word']
                if san_word not in node_index:
                    idx = len(self.nodes)
                    node_index[san_word] = idx
                    self.nodes.append({
                        'word': san_word,
                        'lang': 'san',
                        'lang_family': 'indo_european'
                    })
                entry_nodes.append(node_index[san_word])
            
            # Add cognate nodes
            if 'cognates' in entry:
                for lang, info in entry['cognates'].items():
                    word = info['word']
                    if word not in node_index:
                        idx = len(self.nodes)
                        node_index[word] = idx
                        self.nodes.append({
                            'word': word,
                            'lang': lang,
                            'lang_family': 'indo_european'
                        })
                    entry_nodes.append(node_index[word])
            
            # Create edges (fully connected within cognate set)
            for i in range(len(entry_nodes)):
                for j in range(i + 1, len(entry_nodes)):
                    self.edges.append((entry_nodes[i], entry_nodes[j]))
                    self.edges.append((entry_nodes[j], entry_nodes[i]))  # Undirected
    
    def __len__(self) -> int:
        return 1  # Single graph
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Return the full graph as PyTorch Geometric Data object.
        
        Note: This creates the graph structure. Features should be
        computed using a separate phonetic embedding model.
        """
        import torch
        
        # For now, return raw structure
        # Features will be computed by phonetic embedding model
        edge_index = torch.tensor(self.edges, dtype=torch.long).t()
        
        return {
            'num_nodes': len(self.nodes),
            'edge_index': edge_index,
            'nodes': self.nodes  # Raw node data for feature extraction
        }


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.phonetic_converter import PhoneticConverter
    
    # Test CognateDataset
    converter = PhoneticConverter()
    dataset = CognateDataset(
        data_path="data/raw/sample_etymology_data.json",
        phonetic_converter=converter,
        mode="triplet"
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Anchor: {sample['anchor_text']}, shape: {sample['anchor'].shape}")
    print(f"Positive: {sample['positive_text']}")
    print(f"Negative: {sample['negative_text']}")
