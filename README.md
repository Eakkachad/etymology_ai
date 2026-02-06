# Neural Phonetic Mapping & Etymological Discovery

**ระบบวิเคราะห์และทำนายความสัมพันธ์ทางรากศัพท์ระหว่างภาษาไทย (คำยืมบาลี-สันสกฤต) กับตระกูลภาษาอินโด-ยูโรเปียน**

> *A Deep Learning System for Predicting Etymological Relationships Between Thai (Pali-Sanskrit Loanwords) and Indo-European Language Families*

---

## 🎯 Project Vision

This research project aims to create not just a database, but a **predictive computational tool** that can:

1. **Trace etymological lineages** from Thai loanwords back through Sanskrit/Pali → Proto-Indo-European (PIE) → modern European languages
2. **Predict missing links** in the etymological chain using phonetic similarity and semantic relationships
3. **Visualize language evolution** through interactive graphs and phonetic transformation animations

---

## 🧬 Linguistic Pipeline

```
Thai Loanword (มารดา "mother")
    ↓
Sanskrit (mātṛ)
    ↓
Proto-Indo-European (*méh₂tēr)
    ↓ ↓ ↓
Latin (māter) | Greek (μήτηρ) | English (mother)
```

---

## 🏗️ Technical Architecture

### 1. Phonetic Embedding Layer
- Convert all words to **International Phonetic Alphabet (IPA)**
- Transformer-based encoder to create phonetic vector space
- Extract articulatory features (plosive, aspiration, place of articulation)

### 2. Cognate Prediction (Siamese Network)
- Twin neural networks with shared weights
- **Triplet Loss** to cluster cognates in latent space
- Output: Probability that two words share a common ancestor

### 3. Graph Neural Networks (GNN)
- Nodes = words across all languages
- Edges = known etymological relationships
- **Link Prediction** to discover missing connections

---

## 📊 Data Sources

| Source | Purpose |
|--------|---------|
| **Kaikki** (Wiktionary JSON) | Main etymology database with ancestry chains |
| **WOLD** (World Loanword Database) | Curated Thai loanwords from linguists |
| **Starling** (Tower of Babel) | Deep PIE connections for Sanskrit |
| **PanLex** | Cross-language translation mappings |

---

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate
cd /home/67070309/eak_project/etymology_ai

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate etymology

# Or use pip
pip install -r requirements.txt
```

### Download Sample Data

```bash
# Download Thai etymology data from Kaikki
python scripts/download_sample_data.py --source kaikki --language thai --limit 5000

# Extract Thai-Sanskrit loanwords from WOLD
python scripts/download_sample_data.py --source wold --language thai
```

### Run Phonetic Conversion Demo

```bash
jupyter notebook notebooks/01_phonetic_exploration.ipynb
```

Test with example words:
- **มารดา** (mother) → IPA: /mɑː.dɑː/
- **ไตร** (three) → IPA: /traj/
- **ทศ** (ten) → IPA: /tʰot/

---

## 📁 Project Structure

```
etymology_ai/
├── data/
│   ├── raw/              # Downloaded datasets (Kaikki, WOLD, etc.)
│   ├── processed/        # Cleaned and IPA-converted data
│   └── embeddings/       # Cached phonetic embeddings
├── models/
│   ├── phonetic/         # Phonetic embedding checkpoints
│   ├── cognate/          # Siamese network weights
│   └── gnn/              # Graph neural network models
├── src/
│   ├── data/             # Data downloaders and preprocessors
│   ├── models/           # Model architectures
│   ├── training/         # Multi-GPU training scripts
│   ├── inference/        # Prediction and link discovery
│   └── visualization/    # Graph rendering and animations
├── notebooks/            # Jupyter exploration notebooks
├── tests/                # Unit and integration tests
├── configs/              # YAML configuration files
├── scripts/              # Executable training/inference scripts
└── outputs/
    ├── graphs/           # Interactive etymology graphs
    ├── animations/       # Phonetic evolution videos
    └── reports/          # Research findings
```

---

## 🎯 Project Phases

### ✅ Phase 1: Data Collection (Current)
- Set up infrastructure
- Download and process linguistic datasets
- Build etymology extraction pipeline

### 🔄 Phase 2: Phonetic Normalization (Next)
- Convert all languages to IPA
- Extract phonetic features
- Create cognate pair datasets

### 🔜 Phase 3: Model Development
- Train phonetic embedding layer
- Build Siamese network for cognate detection
- Implement GNN for phylogenetic graphs

### 🔜 Phase 4: DGX A100 Training
- Multi-GPU distributed training
- Synthetic sound change simulation
- Large-scale experimentation

### 🔜 Phase 5: Deployment
- Interactive web dashboard
- REST API for etymology queries
- Visualization tools

---

## 💻 Hardware Requirements

- **Optimal**: DGX A100 (8x GPUs) for large-scale training
- **Minimum**: Single GPU with 16GB+ VRAM for inference and small experiments
- **CPU**: 32+ cores recommended for data preprocessing

---

## 📚 Research References

Key linguistic concepts:
- **Cognates**: Words in different languages with shared ancestry (e.g., "mother" and "mātṛ")
- **Sound Laws**: Regular phonetic changes over time (e.g., Grimm's Law)
- **PIE Reconstruction**: Working backward to hypothetical Proto-Indo-European roots

---

## 🤝 Contributing

This is a research project. Contributions welcome for:
- Additional language data sources
- Improved IPA conversion for low-resource languages
- Novel neural architectures for etymology prediction
- Visualization enhancements

---

## 📄 License

Research project - to be determined based on data source licensing.

---

**Status**: 🚧 Phase 1 - Active Development
