# Etymology AI - Complete User Manual & DGX Deployment Guide

**ระบบวิเคราะห์และทำนายความสัมพันธ์ทางรากศัพท์ด้วย Deep Learning**

---

## 📚 สารบัญ

1. [ภาพรวมระบบ](#1-ภาพรวมระบบ)
2. [การติดตั้งและเตรียมพร้อม](#2-การติดตั้งและเตรียมพร้อม)
3. [การเตรียมข้อมูล](#3-การเตรียมข้อมูล)
4. [การเทรนโมเดล](#4-การเทรนโมเดล)
5. [การใช้งานบน DGX](#5-การใช้งานบน-dgx)
6. [การใช้งานโมเดลที่เทรนแล้ว](#6-การใช้งานโมเดลที่เทรนแล้ว)
7. [การแก้ไขปัญหา](#7-การแก้ไขปัญหา)

---

## 1. ภาพรวมระบบ

### 1.1 จุดประสงค์
ระบบนี้ใช้ Deep Learning ในการ:
- **เรียนรู้ phonetic embeddings** จากคำในหลายภาษา
- **ตรวจจับ cognates** (คำที่มีรากศัพท์เดียวกัน)
- **สร้างกราฟความสัมพันธ์** ทางรากศัพท์

### 1.2 สถาปัตรรรม
```
[Input Words] → [IPA Conversion] → [Phonetic Embedding]
                                          ↓
                                    [Siamese Network] → [Cognate Detection]
                                          ↓
                                    [Etymology GNN] → [Relationship Graph]
```

### 1.3 โมเดลหลัก
1. **Phonetic Embedding Model**: Transformer สำหรับแปลง IPA เป็น vector
2. **Siamese Network**: ตรวจจับคำที่เป็น cognates ด้วย triplet loss
3. **Etymology GNN**: Graph Attention Network สำหรับกราฟความสัมพันธ์

---

## 2. การติดตั้งและเตรียมพร้อม

### 2.1 ความต้องการของระบบ

**สำหรับการพัฒนา (Local)**:
- Python 3.9+ (แนะนำ 3.10)
- GPU: NVIDIA GPU with 16GB+ VRAM
- RAM: 32GB+
- Disk: 50GB free space

**สำหรับการเทรนบน DGX**:
- DGX A100 (8x GPUs)
- SLURM workload manager
- Shared filesystem

### 2.2 การติดตั้ง Dependencies

```bash
cd /home/67070309/eak_project/etymology_ai

# สร้าง Conda environment (แนะนำ)
conda create -n etymology python=3.10
conda activate etymology

# ติดตั้ง PyTorch (CUDA 11.8)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# ติดตั้ง torch-geometric
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# ติดตั้ง dependencies อื่นๆ
pip install pytorch-lightning wandb tensorboard
pip install pythainlp epitran panphon
pip install pandas numpy pyyaml tqdm requests beautifulsoup4
```

### 2.3 โครงสร้างโปรเจกต์

```
etymology_ai/
├── configs/                    # ไฟล์ตั้งค่า
│   ├── data_sources.yaml       
│   └── model_config.yaml       
├── data/                       
│   ├── raw/                    # ข้อมูลดิบ
│   └── processed/              # ข้อมูลที่ประมวณผลแล้ว
├── src/
│   ├── data/                   # โค้ดจัดการข้อมูล
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   ├── phonetic_converter.py
│   │   └── preprocessor.py
│   ├── models/                 # โมเดล Neural Network
│   │   ├── phonetic_embedding.py
│   │   ├── siamese_network.py
│   │   └── etymology_gnn.py
│   ├── training/               # สคริปต์เทรน
│   │   ├── train_phonetic_embedding.py
│   │   └── train_siamese.py
│   └── inference/              # โค้ด inference
│       └── cognate_predictor.py
├── scripts/
│   └── slurm/                  # SLURM job scripts สำหรับ DGX
│       ├── train_phonetic.slurm
│       └── train_siamese.slurm
└── outputs/                    # Output และ checkpoints
```

---

## 3. การเตรียมข้อมูล

### 3.1 สร้างข้อมูลตัวอย่าง

```bash
# สร้าง sample data (สำหรับทดสอบ)
python src/data/sample_dataset.py
```

### 3.2 ดาวน์โหลดข้อมูลจริง (Optional)

```bash
# ดาวน์โหลด Kaikki Thai etymology data
python scripts/download_sample_data.py --source kaikki --language thai

# ประมวลผลข้อมูล
python src/data/preprocessor.py
```

### 3.3 ตรวจสอบข้อมูล

```python
from src.data.dataset import CognateDataset
from src.data.phonetic_converter import PhoneticConverter

converter = PhoneticConverter()
dataset = CognateDataset(
    "data/raw/sample_etymology_data.json",
    converter,
    mode="triplet"
)

print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample: {sample['anchor_text']} - {sample['positive_text']}")
```

---

## 4. การเทรนโมเดล

### 4.1 เทรน Phonetic Embedding (ขั้นที่ 1)

**Local (Single GPU)**:
```bash
python src/training/train_phonetic_embedding.py \
    --config configs/model_config.yaml \
    --data data/raw/sample_etymology_data.json \
    --output outputs/phonetic_embedding \
    --epochs 50 \
    --batch-size 32
```

**DGX (8 GPUs)** - ดูส่วนที่ 5

### 4.2 เทรน Siamese Network (ขั้นที่ 2)

```bash
python src/training/train_siamese.py \
    --config configs/model_config.yaml \
    --data data/raw/sample_etymology_data.json \
    --encoder outputs/phonetic_embedding/checkpoints/best.ckpt \
    --output outputs/siamese
```

### 4.3 ติดตามการเทรน

```bash
# เปิด TensorBoard
tensorboard --logdir outputs/

# เปิดเบราว์เซอร์ไปที่ http://localhost:6006
```

---

## 5. การใช้งานบน DGX

### 5.1 เชื่อมต่อกับ DGX

```bash
# SSH เข้า DGX
ssh username@dgx-server-ip

# ตรวจสอบ GPUs
nvidia-smi

# ตรวจสอบ SLURM
sinfo
```

### 5.2 เตรียม Environment บน DGX

```bash
# โคลนโปรเจกต์
cd /shared/username/
git clone https://github.com/Eakkachad/etymology_ai.git
cd etymology_ai

# สร้าง Conda environment
conda create -n etymology python=3.10
conda activate etymology

# ติดตั้ง dependencies (ตามขั้นตอนในส่วน 2.2)
```

### 5.3 Submit SLURM Jobs

#### เทรน Phonetic Embedding

```bash
# แก้ไข train_phonetic.slurm ตามความต้องการ
nano scripts/slurm/train_phonetic.slurm

# Submit job
sbatch scripts/slurm/train_phonetic.slurm

# ตรวจสอบสถานะ
squeue -u $USER

# ดู log
tail -f outputs/logs/phonetic_*.out
```

#### เทรน Siamese Network

```bash
# หลังจาก phonetic embedding เทรนเสร็จ
sbatch scripts/slurm/train_siamese.slurm
```

### 5.4 การจัดการ Jobs

```bash
# ดูคิวงาน
squeue

# ยกเลิก job
scancel <job_id>

# ดูรายละเอียด job
scontrol show job <job_id>

# ดูประวัติ
sacct -u $USER
```

### 5.5 การตั้งค่า Multi-GPU

โมเดลใช้ PyTorch Lightning DDP อัตโนมัติ:
- `devices=8` ใน config = ใช้ 8 GPUs
- `strategy="ddp"` = Distributed Data Parallel
- Automatic gradient synchronization
- Linear scaling of batch size

---

## 6. การใช้งานโมเดลที่เทรนแล้ว

### 6.1 Cognate Detection

```python
from src.inference.cognate_predictor import CognatePredictor

# Load trained model
predictor = CognatePredictor('outputs/siamese/checkpoints/best.ckpt')

# Predict cognate score
score = predictor.predict(
    word1="mātṛ",   lang1="san",
    word2="mother", lang2="eng"
)
print(f"Cognate probability: {score:.3f}")

# Find cognates from candidate list
candidates = ["mother", "father", "brother", "water", "fire"]
cognates = predictor.find_cognates(
    query_word="mātṛ",
    query_lang="san",
    candidate_words=candidates,
    candidate_lang="eng",
    threshold=0.7
)
print(f"Cognates: {cognates}")
```

### 6.2 Phonetic Embedding Extraction

```python
from src.models.phonetic_embedding import PhoneticEmbeddingModel
import torch

model = PhoneticEmbeddingModel.load_from_checkpoint(
    'outputs/phonetic_embedding/checkpoints/best.ckpt'
)
model.eval()

# Encode word
word_ipa = "maːtr̩"
codes = [ord(c) for c in word_ipa[:50]]
codes += [0] * (50 - len(codes))
x = torch.tensor([codes], dtype=torch.long)

with torch.no_grad():
    embedding = model(x)

print(f"Embedding shape: {embedding.shape}")  # (1, 512)
```

---

## 7. การแก้ไขปัญหา

### 7.1 Out of Memory (OOM)

**ปัญหา**: GPU memory เต็ม

**วิธีแก้**:
```bash
# ลด batch size
python src/training/train_phonetic_embedding.py --batch-size 128

# หรือเพิ่ม gradient accumulation
# แก้ใน configs/model_config.yaml:
# accumulate_grad_batches: 4
```

### 7.2 SLURM Job Failed

```bash
# ดู error log
cat outputs/logs/phonetic_<job_id>.err

# ปัญหาทั่วไป:
# 1. CUDA version mismatch → แก้ module load cuda 
# 2. Conda environment ไม่ active → ตรวจสอบ SLURM script
# 3. Path ไม่ถูกต้อง → ใช้ absolute paths
```

### 7.3 pythainlp ใช้งานไม่ได้

```bash
# ต้องการ Python 3.9+
conda install python=3.10

# ติดตั้ง pythainlp ใหม่
pip install pythainlp==5.0.0
```

### 7.4 torch-geometric ติดตั้งไม่ได้

```bash
# ใช้ conda (ง่ายกว่า)
conda install pyg -c pyg

# หรือ pip แบบระบุ CUDA version
pip install torch-geometric torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

---

## 8. Performance Tuning

###8.1 Multi-GPU Scaling

**Expected speedup**:
- 1 GPU: ~100 samples/sec
- 4 GPUs: ~380 samples/sec (3.8x)
- 8 GPUs: ~720 samples/sec (7.2x)

**Tips**:
- ใช้ `num_workers=4` ใน DataLoader
- Enable `pin_memory=True`
- ใช้ `precision="16-mixed"` สำหรับ mixed precision training

### 8.2 Monitoring

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training progress
tensorboard --logdir outputs/

# System resources
htop
```

---

## 9. คำถามที่พบบ่อย (FAQ)

**Q: ต้องเทรนตามลำดับหรือไม่?**
A: ใช่ ต้องเทรนตามลำดับ: Phonetic Embedding → Siamese Network → GNN

**Q: ใช้เวลาเทรนนานแค่ไหน?**
A: บน DGX A100 (8 GPUs):
- Phonetic Embedding: ~3-4 hours (100 epochs)
- Siamese Network: ~2-3 hours

**Q: สามารถใช้ CPU เทรนได้หรือไม่?**
A: ได้แต่ช้ามาก (~50-100x) ไม่แนะนำสำหรับ dataset ขนาดใหญ่

**Q: checkpoint files ใหญ่แค่ไหน?**
A:
- Phonetic Embedding: ~200-300 MB
- Siamese Network: ~250-350 MB
- GNN: ~150-200 MB

---

## 10. การพัฒนาต่อ

### 10.1 Adding New Languages

1. เพิ่ม config ใน `configs/data_sources.yaml`
2. Implement IPA conversion ใน `phonetic_converter.py`
3. เพิ่ม language family encoding

### 10.2 Custom Models

ดูตัวอย่างใน `src/models/` และสร้าง class ใหม่ที่สืบทอดจาก `nn.Module`

### 10.3 Hyperparameter Tuning

แก้ไข configs/model_config.yaml`:
- learning_rate
- batch_size
- num_layers, num_heads
- dropout

---

ผู้พัฒนา: Eakkachad & Antigravity AI Assistant
อัพเดทล่าสุด: 2026-02-07
