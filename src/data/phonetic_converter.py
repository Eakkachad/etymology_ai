"""
Phonetic conversion utilities for multi-language IPA transformation.

Supports:
- Thai → IPA (PyThaiNLP)
- Sanskrit/Pali → IPA (custom transliteration)
- Latin/Greek → IPA (Epitran)
- Phonetic feature extraction
"""

from typing import Dict, List, Tuple
import re
import logging

try:
    from pythainlp import transliterate
    from pythainlp.tokenize import word_tokenize
except ImportError:
    logging.warning("PyThaiNLP not installed. Thai IPA conversion will be limited.")

try:
    import epitran
except ImportError:
    logging.warning("Epitran not installed. Some IPA conversions will fail.")

try:
    import panphon
except ImportError:
    logging.warning("Panphon not installed. Phonetic feature extraction unavailable.")


logger = logging.getLogger(__name__)


class PhoneticConverter:
    """Convert words from various languages to IPA representation."""
    
    def __init__(self):
        """Initialize converters for supported languages."""
        self.epitran_converters = {}
        
        # Initialize Epitran for supported languages
        try:
            self.epitran_converters['lat'] = epitran.Epitran('lat-Latn')  # Latin
            self.epitran_converters['eng'] = epitran.Epitran('eng-Latn')  # English
        except:
            logger.warning("Could not initialize Epitran converters")
        
        # Initialize Panphon for feature extraction
        try:
            self.feature_tool = panphon.FeatureTable()
        except:
            self.feature_tool = None
    
    def to_ipa(self, word: str, language: str) -> str:
        """
        Convert word to IPA based on language.
        
        Args:
            word: Word to convert
            language: Language code ('tha', 'san', 'lat', 'eng', etc.)
        
        Returns:
            IPA representation
        """
        if language == 'tha':
            return self._thai_to_ipa(word)
        elif language in ['san', 'pli']:
            return self._sanskrit_to_ipa(word)
        elif language in self.epitran_converters:
            return self.epitran_converters[language].transliterate(word)
        else:
            logger.warning(f"No IPA converter for language: {language}")
            return word
    
    def _thai_to_ipa(self, word: str) -> str:
        """
        Convert Thai word to IPA.
        
        Uses PyThaiNLP's romanization as intermediate step.
        """
        try:
            # Use Royal Thai Transcription as base
            romanized = transliterate.romanize(word, engine='royin')
            
            # Manual mapping for common Thai sounds
            # This is a simplified version - production should use phonological rules
            ipa_map = {
                'ch': 'tɕ',
                'th': 'tʰ',
                'ph': 'pʰ',
                'kh': 'kʰ',
                'ng': 'ŋ',
                'a': 'aː',
                'i': 'iː',
                'u': 'uː',
                'e': 'eː',
                'o': 'oː',
            }
            
            ipa = romanized.lower()
            for rom, ipa_char in sorted(ipa_map.items(), key=lambda x: -len(x[0])):
                ipa = ipa.replace(rom, ipa_char)
            
            return ipa
        
        except:
            logger.warning(f"Failed to convert Thai word: {word}")
            return word
    
    def _sanskrit_to_ipa(self, word: str) -> str:
        """
        Convert Sanskrit/Pali (in romanized form) to IPA.
        
        Assumes IAST (International Alphabet of Sanskrit Transliteration).
        """
        # Sanskrit IAST to IPA mapping
        iast_to_ipa = {
            # Vowels
            'ā': 'aː',
            'ī': 'iː',
            'ū': 'uː',
            'ṛ': 'r̩',
            'ṝ': 'r̩ː',
            'ḷ': 'l̩',
            'ḹ': 'l̩ː',
            'ē': 'eː',
            'ai': 'ai̯',
            'ō': 'oː',
            'au': 'au̯',
            
            # Consonants (retroflex)
            'ṭ': 'ʈ',
            'ṭh': 'ʈʰ',
            'ḍ': 'ɖ',
            'ḍh': 'ɖʱ',
            'ṇ': 'ɳ',
            
            # Palatals
            'c': 'tɕ',
            'ch': 'tɕʰ',
            'j': 'dʑ',
            'jh': 'dʑʱ',
            'ñ': 'ɲ',
            
            # Velars
            'k': 'k',
            'kh': 'kʰ',
            'g': 'ɡ',
            'gh': 'ɡʱ',
            'ṅ': 'ŋ',
            
            # Special
            'ś': 'ɕ',
            'ṣ': 'ʂ',
            'ḥ': 'ɦ',
            'ṃ': 'ŋ',
        }
        
        ipa = word.lower()
        # Sort by length to replace longer patterns first
        for iast, ipa_char in sorted(iast_to_ipa.items(), key=lambda x: -len(x[0])):
            ipa = ipa.replace(iast, ipa_char)
        
        return ipa
    
    def extract_features(self, ipa_string: str) -> List[Dict]:
        """
        Extract phonetic features for each phoneme.
        
        Returns:
            List of feature dictionaries for each phoneme
        """
        if not self.feature_tool:
            logger.warning("Panphon not available for feature extraction")
            return []
        
        features = []
        for segment in self.feature_tool.ipa_segs(ipa_string):
            feature_vector = self.feature_tool.fts(segment)
            features.append({
                'phoneme': segment,
                'features': feature_vector.numeric()
            })
        
        return features
    
    def phonetic_distance(self, ipa1: str, ipa2: str) -> float:
        """
        Calculate phonetic distance between two IPA strings.
        
        Uses feature-weighted edit distance.
        """
        if not self.feature_tool:
            # Fallback to simple Levenshtein
            return self._levenshtein(ipa1, ipa2)
        
        try:
            # Use Panphon's feature edit distance
            distance = self.feature_tool.feature_edit_distance(ipa1, ipa2)
            return distance
        except:
            return self._levenshtein(ipa1, ipa2)
    
    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        """Simple Levenshtein distance implementation."""
        if len(s1) < len(s2):
            return PhoneticConverter._levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


# Example usage
if __name__ == "__main__":
    converter = PhoneticConverter()
    
    # Test Thai conversion
    thai_words = ["มารดา", "ไตร", "ทศ"]
    print("Thai → IPA:")
    for word in thai_words:
        ipa = converter.to_ipa(word, 'tha')
        print(f"  {word} → {ipa}")
    
    # Test Sanskrit conversion
    sanskrit_words = ["mātṛ", "tri", "daśa"]
    print("\nSanskrit → IPA:")
    for word in sanskrit_words:
        ipa = converter.to_ipa(word, 'san')
        print(f"  {word} → {ipa}")
        
        # Extract features
        features = converter.extract_features(ipa)
        if features:
            print(f"    Features: {features[0]}")
    
    # Test phonetic distance
    print("\nPhonetic Distances:")
    print(f"  tri vs. three: {converter.phonetic_distance('tri', 'θriː')}")
    print(f"  daśa vs. decem: {converter.phonetic_distance('daɕa', 'dekem')}")
