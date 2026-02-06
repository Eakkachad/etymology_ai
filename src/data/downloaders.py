"""
Data downloaders for linguistic datasets.

This module provides classes to download and cache data from:
- Kaikki (Wiktionary JSON)
- WOLD (World Loanword Database)
- Starling (Tower of Babel PIE database)
- PanLex (Cross-language translations)
"""

import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDownloader:
    """Base class for all data downloaders."""
    
    def __init__(self, config_path: str = "configs/data_sources.yaml"):
        """Initialize downloader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.cache_dir = Path("data/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192) -> None:
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {output_path}")
        
        # Decompress if gzip
        if output_path.suffix == '.gz':
            import gzip
            import shutil
            
            decompressed_path = output_path.with_suffix('')
            logger.info(f"Decompressing to: {decompressed_path}")
            
            with gzip.open(output_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Decompressed: {decompressed_path}")
            return decompressed_path
        
        return output_path


class KaikkiDownloader(BaseDownloader):
    """Download Wiktionary etymology data from Kaikki."""
    
    def download_language(self, language: str = "Thai") -> Path:
        """
        Download Kaikki data for a specific language.
        
        Args:
            language: Language name (e.g., "Thai", "Sanskrit")
        
        Returns:
            Path to downloaded JSON file (decompressed if was .gz)
        """
        lang_key = language.lower()
        
        if lang_key not in self.config['kaikki']['languages']:
            raise ValueError(f"Language {language} not configured in data_sources.yaml")
        
        url = self.config['kaikki']['languages'][lang_key]['url']
        
        # Determine output path (handle .gz extension)
        if url.endswith('.gz'):
            output_path = self.cache_dir / f"kaikki_{lang_key}.jsonl.gz"
            decompressed_path = self.cache_dir / f"kaikki_{lang_key}.jsonl"
        else:
            output_path = self.cache_dir / f"kaikki_{lang_key}.jsonl"
            decompressed_path = output_path
        
        # Check if decompressed file exists
        if decompressed_path.exists():
            logger.info(f"Cache exists: {decompressed_path}")
            return decompressed_path
        
        logger.info(f"Downloading Kaikki {language} data...")
        result_path = self.download_file(url, output_path)
        
        # download_file returns decompressed path if it was .gz
        return result_path if result_path else output_path
    
    def load_etymology_data(self, language: str = "Thai") -> List[Dict]:
        """
        Load and parse etymology entries.
        
        Returns:
            List of entries with etymology information
        """
        file_path = self.download_language(language)
        
        entries_with_etymology = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Parsing {language}"):
                try:
                    entry = json.loads(line.strip())
                    
                    # Filter entries with etymology information
                    if 'etymology_text' in entry or 'etymology_templates' in entry:
                        entries_with_etymology.append(entry)
                
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Found {len(entries_with_etymology)} entries with etymology")
        return entries_with_etymology


class WOLDDownloader(BaseDownloader):
    """Download Thai loanword data from World Loanword Database."""
    
    def download_thai_loanwords(self) -> Path:
        """
        Download Thai language data with loanword annotations.
        
        Returns:
            Path to cached data directory
        """
        wold_config = self.config['wold']
        cache_dir = Path(wold_config['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Fetching WOLD Thai loanwords...")
        
        # Note: WOLD may require web scraping or API access
        # Placeholder implementation
        output_file = cache_dir / "thai_loanwords.json"
        
        if output_file.exists():
            logger.info(f"Cache exists: {output_file}")
            return output_file
        
        # TODO: Implement actual WOLD scraping/API logic
        logger.warning("WOLD download not fully implemented - requires web scraping")
        
        return output_file


class StarlingDownloader(BaseDownloader):
    """Download PIE etymology data from Starling database."""
    
    def download_pie_roots(self) -> Path:
        """
        Download Proto-Indo-European root database.
        
        Returns:
            Path to PIE roots file
        """
        starling_config = self.config['starling']
        cache_dir = Path(starling_config['cache_dir'])
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = cache_dir / "pie_roots.html"
        
        if output_file.exists():
            logger.info(f"Cache exists: {output_file}")
            return output_file
        
        # TODO: Implement Starling database scraping
        logger.warning("Starling download not fully implemented - may require manual download")
        
        return output_file


class PanLexDownloader(BaseDownloader):
    """Download cross-language translation data from PanLex API."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url = self.config['panlex']['api_url']
        self.rate_limit = self.config['panlex']['rate_limit']
    
    def query_translations(self, word: str, source_lang: str = "tha") -> List[Dict]:
        """
        Query PanLex for translations of a word.
        
        Args:
            word: Word to translate
            source_lang: Source language code (e.g., "tha" for Thai)
        
        Returns:
            List of translation entries
        """
        endpoint = f"{self.api_url}/expr"
        
        params = {
            'txt': word,
            'uid': source_lang
        }
        
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        
        return response.json().get('result', [])


# Example usage
if __name__ == "__main__":
    # Download Thai etymology data
    kaikki = KaikkiDownloader()
    thai_data = kaikki.load_etymology_data("Thai")
    
    print(f"\nSample Thai entry:")
    print(json.dumps(thai_data[0], indent=2, ensure_ascii=False))
    
    # Download Sanskrit for comparison
    sanskrit_data = kaikki.load_etymology_data("Sanskrit")
    print(f"\nTotal Sanskrit entries with etymology: {len(sanskrit_data)}")
