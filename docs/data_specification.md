# Data Specification Document

## Overview

This document details the exact data requirements for the Etymology AI project, including data types, quantities, formats, and processing pipeline.

---

## 1. Kaikki (Wiktionary Extracted Data)

### Description
Pre-processed Wiktionary dumps in structured JSON format, extracted by the Kaikki project.

### Languages & Expected Sizes

| Language | File Size | # Entries (Est.) | Purpose |
|----------|-----------|------------------|---------|
| Thai | ~50 MB | ~30,000 | Source language (loanwords) |
| Sanskrit | ~200 MB | ~100,000 | Intermediate layer (PIE branch) |
| Pali | ~20 MB | ~15,000 | Buddhist texts influence |
| Latin | ~400 MB | ~300,000 | Classical IE comparison |
| Ancient Greek | ~300 MB | ~200,000 | Classical IE comparison |
| English | ~2.5 GB | ~1,000,000 | Modern IE endpoint |

### Format Structure

```json
{
  "word": "มารดา",
  "lang": "Thai",
  "lang_code": "th",
  "pos": "noun",
  "etymology_text": "From Pali mātā, from Sanskrit mātṛ (\"mother\")",
  "etymology_templates": [
    {
      "name": "bor",
      "args": {
        "1": "th",
        "2": "pi",
        "3": "mātā"
      },
      "expansion": "Borrowed from Pali mātā"
    }
  ],
  "senses": [
    {
      "glosses": ["mother"],
      "id": "th-มารดา-noun-12345"
    }
  ]
}
```

### What We Extract
- `word`: The headword
- `etymology_text`: Free-form etymology description
- `etymology_templates`: Structured etymology links
- `senses`: Meanings for semantic similarity

### Processing Pipeline
1. Download JSONL files (one JSON object per line)
2. Filter entries with `etymology_templates` or `etymology_text`
3. Extract loanword chains: `Thai → Pali/Sanskrit → PIE`
4. Store in PostgreSQL with full-text search enabled

---

## 2. WOLD (World Loanword Database)

### Description
Curated dataset of loanwords across 41 languages by linguists at Max Planck Institute.

### Thai Language Data

| Metric | Value |
|--------|-------|
| Total vocabulary entries | ~1,460 meanings |
| Loanwords from Sanskrit/Pali | ~100-150 words |
| Format | CSV / TSV export from database |
| License | CC BY 4.0 |

### Format Structure

```csv
word_id,language,word_form,meaning,source_language,borrowed_score,cognacy_class
1234,Thai,มารดา,mother,Pali,1.0,IE-3456
1235,Thai,ไตร,three,Sanskrit,1.0,IE-7890
```

### What We Extract
- Confirmed loanword pairs (Thai ↔ Sanskrit/Pali)
- Cognacy class IDs (links to PIE roots)
- Borrowed confidence scores (gold standard for training)

### Processing Pipeline
1. Download Thai vocabulary export
2. Filter by `source_language IN ('Sanskrit', 'Pali')`
3. Use as **ground truth** for cognate detection training
4. Cross-reference with Kaikki data for IPA conversion

---

## 3. Starling (Tower of Babel)

### Description
Comprehensive etymological database for major language families, created by linguists Sergei Starostin and George Starostin.

### Datasets Needed

| Database | Size | Entries | Format |
|----------|------|---------|--------|
| Indo-European | ~5 MB HTML | ~3,000 PIE roots | Web scraping required |
| Sanskrit roots | ~2 MB HTML | ~1,500 roots | Web scraping required |

### Format Structure (After Scraping)

```json
{
  "pie_root": "*méh₂tēr",
  "meaning": "mother",
  "descendants": [
    {"language": "Sanskrit", "form": "mātṛ"},
    {"language": "Latin", "form": "māter"},
    {"language": "Ancient Greek", "form": "μήτηρ"},
    {"language": "English", "form": "mother"}
  ],
  "sound_laws": [
    "PIE *h₂ > Sanskrit /ā/",
    "PIE *ē > Latin /ā/"
  ]
}
```

### Processing Pipeline
1. Web scrape from `starling.rinet.ru`
2. Parse HTML tables to extract PIE roots
3. Build graph: `PIE root → Language branches`
4. Store in graph database (Neo4j or NetworkX)

---

## 4. PanLex (Translation Database)

### Description
Massive cross-language translation database with 1,353 language varieties.

### API Limits & Strategy

| Metric | Value |
|--------|-------|
| API rate limit | 10 requests/second |
| Expected queries | ~5,000 words × 6 languages = 30,000 queries |
| Estimated time | ~50 minutes |
| Format | JSON via REST API |

### Sample API Response

```json
{
  "result": [
    {
      "expr_id": "12345678",
      "txt": "มารดา",
      "lang": "tha"
    }
  ],
  "translations": [
    {
      "lang": "san",
      "txt": "मातृ",
      "quality": 9
    },
    {
      "lang": "eng",
      "txt": "mother",
      "quality": 10
    }
  ]
}
```

### What We Extract
- Cross-language word alignments
- Translation quality scores
- Use for **semantic similarity** validation

### Processing Pipeline
1. Query for each Thai-Sanskrit loanword pair
2. Fetch translations to English/Latin/Greek
3. Build translation graph for semantic clustering
4. Cache locally to avoid re-querying

---

## 5. Swadesh Lists (Test Dataset)

### Description
Standard vocabulary lists (100-207 words) used in historical linguistics for comparison.

### Why Use This?

- **Highly conserved**: Basic words less likely to be replaced by loanwords
- **Good cognates**: Many PIE cognates still visible
- **Small scope**: Perfect for initial model testing

### Format

| Word # | English | Thai | Sanskrit | Latin | Greek |
|--------|---------|------|----------|-------|-------|
| 1 | I | ฉัน | aham | ego | ἐγώ |
| 2 | you | คุณ | tvam | tū | σύ |
| 3 | mother | มารดา | mātṛ | māter | μήτηρ |

### Processing
- Start with Swadesh 100 for Phase 2 IPA testing
- Expand to full etymology dataset in Phase 3

---

## Total Data Volume Summary

| Data Source | Raw Size | Processed Size | Priority |
|-------------|----------|----------------|----------|
| Kaikki (6 languages) | ~3.5 GB | ~500 MB (filtered) | **High** |
| WOLD Thai | ~5 MB | ~2 MB | **Critical** |
| Starling PIE | ~10 MB | ~5 MB (structured) | **High** |
| PanLex (cached) | N/A | ~50 MB | **Medium** |
| Swadesh lists | ~1 MB | ~500 KB | **High** |
| **Total** | **~3.5 GB** | **~550 MB** | - |

---

## Storage & Database Strategy

### File Storage
```
data/
├── raw/                          # Original downloads (3.5 GB)
│   ├── kaikki/
│   │   ├── thai.jsonl           # 50 MB
│   │   ├── sanskrit.jsonl       # 200 MB
│   │   └── ...
│   ├── wold/
│   │   └── thai_vocabulary.csv  # 5 MB
│   ├── starling/
│   │   └── pie_roots.json       # 5 MB (after processing)
│   └── panlex/
│       └── cached_queries.json  # 50 MB
│
└── processed/                    # Cleaned & IPA-converted (550 MB)
    ├── etymology_pairs.parquet  # Thai → Sanskrit cognates
    ├── ipa_phonetic.parquet     # All words in IPA format
    └── cognate_graph.graphml    # NetworkX graph export
```

### Database Schema (PostgreSQL)

```sql
-- Words table
CREATE TABLE words (
    word_id SERIAL PRIMARY KEY,
    word_form TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,
    ipa TEXT,
    phonetic_features JSONB,
    meaning TEXT
);

-- Etymology links
CREATE TABLE etymology_links (
    link_id SERIAL PRIMARY KEY,
    source_word_id INT REFERENCES words(word_id),
    target_word_id INT REFERENCES words(word_id),
    relationship VARCHAR(50), -- 'borrowed', 'cognate', 'derived'
    confidence FLOAT
);

-- PIE roots
CREATE TABLE pie_roots (
    root_id SERIAL PRIMARY KEY,
    reconstructed_form TEXT,
    meaning TEXT,
    descendants JSONB
);
```

---

## Next Steps for Data Collection

1. **Phase 1a**: Download Kaikki Thai + Sanskrit (250 MB)
2. **Phase 1b**: Extract WOLD Thai loanwords (5 MB) ← **Gold standard**
3. **Phase 1c**: Scrape Starling PIE roots (10 MB)
4. **Phase 2**: Convert all to IPA and store in database
5. **Phase 3**: Build training datasets for models

---

**Total Disk Space Required**: ~5 GB (with backups and intermediate files)

**Estimated Download Time**: ~30 minutes (depends on connection)

**Processing Time**: ~2-3 hours for full pipeline
