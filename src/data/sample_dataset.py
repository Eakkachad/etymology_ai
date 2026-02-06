"""
Sample etymology dataset for testing.

This manually curated dataset contains confirmed Thai loanwords from Pali/Sanskrit
with their PIE roots and modern cognates. Used for initial model development.
"""

import json
from pathlib import Path


# Curated sample data: Thai → Sanskrit → PIE → Modern languages
SAMPLE_DATA = [
    # Numbers (highly conserved)
    {
        "thai": {"word": "ไตร", "pos": "numeral", "meaning": "three"},
        "sanskrit": {"word": "tri", "devanagari": "त्रि", "ipa": "tri"},
        "pie": {"reconstructed": "*tréyes"},
        "cognates": {
            "latin": {"word": "trēs", "ipa": "treːs"},
            "greek": {"word": "τρεῖς", "romanized": "treîs", "ipa": "trêːs"},
            "english": {"word": "three", "ipa": "θriː"}
        }
    },
    {
        "thai": {"word": "ทศ", "pos": "numeral", "meaning": "ten"},
        "sanskrit": {"word": "daśa", "devanagari": "दश", "ipa": "daɕa"},
        "pie": {"reconstructed": "*deḱm̥"},
        "cognates": {
            "latin": {"word": "decem", "ipa": "dekem"},
            "greek": {"word": "δέκα", "romanized": "déka", "ipa": "deka"},
            "english": {"word": "ten", "ipa": "tɛn"}
        }
    },
    
    # Family terms
    {
        "thai": {"word": "มารดา", "pos": "noun", "meaning": "mother"},
        "sanskrit": {"word": "mātṛ", "devanagari": "मातृ", "ipa": "maːtɨ"},
        "pie": {"reconstructed": "*méh₂tēr"},
        "cognates": {
            "latin": {"word": "māter", "ipa": "maːter"},
            "greek": {"word": "μήτηρ", "romanized": "mētēr", "ipa": "mɛ̌ːtɛːr"},
            "english": {"word": "mother", "ipa": "mʌðər"}
        }
    },
    {
        "thai": {"word": "บิดา", "pos": "noun", "meaning": "father"},
        "sanskrit": {"word": "pitṛ", "devanagari": "पितृ", "ipa": "pitɨ"},
        "pie": {"reconstructed": "*ph₂tḗr"},
        "cognates": {
            "latin": {"word": "pater", "ipa": "pater"},
            "greek": {"word": "πατήρ", "romanized": "patēr", "ipa": "patɛːr"},
            "english": {"word": "father", "ipa": "fɑːðər"}
        }
    },
    
    # Basic verbs/concepts
    {
        "thai": {"word": "ทาน", "pos": "noun", "meaning": "charity, giving"},
        "sanskrit": {"word": "dāna", "devanagari": "दान", "ipa": "daːna"},
        "pie": {"reconstructed": "*deh₃-"},
        "cognates": {
            "latin": {"word": "dōnum", "ipa": "doːnum"},
            "greek": {"word": "δῶρον", "romanized": "dôron", "ipa": "dɔ̂ːron"},
            "english": {"word": "donate", "ipa": "doʊneɪt"}
        }
    },
    {
        "thai": {"word": "นาม", "pos": "noun", "meaning": "name"},
        "sanskrit": {"word": "nāman", "devanagari": "नामन्", "ipa": "naːman"},
        "pie": {"reconstructed": "*h₁nómn̥"},
        "cognates": {
            "latin": {"word": "nōmen", "ipa": "noːmen"},
            "greek": {"word": "ὄνομα", "romanized": "ónoma", "ipa": "ónoma"},
            "english": {"word": "name", "ipa": "neɪm"}
        }
    },
    
    # Body parts
    {
        "thai": {"word": "นัย", "pos": "noun", "meaning": "eye (archaic)"},
        "sanskrit": {"word": "nayana", "devanagari": "नयन", "ipa": "najana"},
        "pie": {"reconstructed": "*h₃ekʷ-"},
        "cognates": {
            "latin": {"word": "oculus", "ipa": "ɔkʊlʊs"},
            "greek": {"word": "ὄσσε", "romanized": "ósse", "ipa": "osse"},
            "english": {"word": "eye", "ipa": "aɪ"}
        }
    },
    
    # Nature
    {
        "thai": {"word": "อัคคี", "pos": "noun", "meaning": "fire"},
        "sanskrit": {"word": "agni", "devanagari": "अग्नि", "ipa": "agni"},
        "pie": {"reconstructed": "*h₁n̥gʷnis"},
        "cognates": {
            "latin": {"word": "ignis", "ipa": "ignis"},
            "english": {"word": "ignite", "ipa": "ɪɡnaɪt"}
        }
    }
]


def create_sample_dataset(output_dir: str = "data/raw"):
    """Create sample JSON dataset for testing."""
    output_path = Path(output_dir) / "sample_etymology_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(SAMPLE_DATA, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Created sample dataset: {output_path}")
    print(f"  Total entries: {len(SAMPLE_DATA)}")
    print(f"  Languages: Thai, Sanskrit, PIE, Latin, Greek, English")
    
    return output_path


if __name__ == "__main__":
    create_sample_dataset()
