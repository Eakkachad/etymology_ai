"""
Inference utilities for trained models.
"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.phonetic_embedding import PhoneticEmbeddingModel
from src.models.siamese_network import SiameseNetwork
from src.data.phonetic_converter import PhoneticConverter


class CognatePredictor:
    """
    Predict whether two words are cognates.
    """
    
    def __init__(
        self,
        model_checkpoint: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.converter = PhoneticConverter()
        
        # Load model
        self.model = torch.load(model_checkpoint, map_location=device)
        self.model.eval()
    
    def _encode_word(self, word: str, lang: str) -> torch.Tensor:
        """Convert word to IPA and encode."""
        ipa = self.converter.to_ipa(word, lang)
        
        # Simple character encoding
        codes = [ord(c) for c in ipa[:50]]
        while len(codes) < 50:
            codes.append(0)
        
        return torch.tensor([codes], dtype=torch.long, device=self.device)
    
    def predict(
        self,
        word1: str,
        lang1: str,
        word2: str,
        lang2: str
    ) -> float:
        """
        Predict cognate probability.
        
        Returns:
            Similarity score (higher = more likely cognates)
        """
        x1 = self._encode_word(word1, lang1)
        x2 = self._encode_word(word2, lang2)
        
        with torch.no_grad():
            similarity = self.model(x1, x2)
        
        return similarity.item()
    
    def find_cognates(
        self,
        query_word: str,
        query_lang: str,
        candidate_words: list,
        candidate_lang: str,
        threshold: float = 0.7
    ) -> list:
        """
        Find cognates from a list of candidates.
        
        Returns:
            List of (word, score) tuples above threshold
        """
        results = []
        
        for candidate in candidate_words:
            score = self.predict(query_word, query_lang, candidate, candidate_lang)
            if score >= threshold:
                results.append((candidate, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


# Example usage
if __name__ == "__main__":
    # This would load a trained model
    # predictor = CognatePredictor('outputs/siamese/checkpoints/best.ckpt')
    
    # Example prediction
    # score = predictor.predict("mātṛ", "san", "mother", "eng")
    # print(f"Cognate score: {score:.3f}")
    
    print("CognatePredictor class defined. Load a trained model to use.")
