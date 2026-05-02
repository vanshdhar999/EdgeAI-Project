# Edge AI Plant Disease Detection System

An offline, edge-deployable plant disease detection system that runs entirely on a **Raspberry Pi 5**. A MobileNetV3-Small model trained on PlantVillage is exported to ONNX and quantized to INT8 for real-time leaf disease classification — no internet required.

**Inference latency on Pi 5:** ~6 ms (INT8) · **Model size:** 1.1 MB · **Classes:** 15 (Tomato × 10, Potato × 3, Pepper × 2)

**Team:** Aayush Jeevan Patil (22220) · Vansh Dhar (22156)  
**Full report:** [`docs/report.md`](docs/report.md)

---

## Project Structure

```
├── src/
│   ├── data_prep.py        # Dataset split, preprocessing, labels.txt
│   ├── augmentation.py     # torchvision augmentation pipeline
│   ├── train.py            # PyTorch two-stage fine-tuning
│   ├── evaluate.py         # Accuracy, confusion matrix, per-class metrics
│   ├── quantize.py         # ONNX export + INT8 static quantization
│   └── benchmark.py        # Latency benchmarking (dev or Pi)
├── deployment/
│   ├── inference.py        # PlantDiseaseClassifier — ONNX Runtime
│   ├── live_camera.py      # Live feed + disease overlay
│   ├── setup_pi.sh         # Pi environment setup script
│   └── requirements_pi.txt # Minimal Pi dependencies
├── models/
│   ├── plant_disease_float32.onnx
│   ├── plant_disease.onnx           # INT8 quantized — use this on Pi
│   └── labels.txt
├── docs/
│   ├── report.md
│   ├── dataset_notes.md
│   └── confusion_matrix.png
├── requirements.txt
└── submission.txt
```

---

## Step-by-Step Reproduction

### Dev Machine (Training)

```bash
git clone https://github.com/vanshdhar999/EdgeAI-Project.git
cd EdgeAI-Project

# Create environment
conda create -n pydl python=3.11
conda activate pydl

# Install PyTorch with GPU (replace cu121 with your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download **PlantVillage** from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract to `plantvillage-dataset/color/`.

```bash
# 1. Prepare dataset (balanced mode — 15 classes, 1000 images/class cap)
conda run -n pydl python3 src/data_prep.py

# 2. Train model (two-stage MobileNetV3-Small fine-tuning)
conda run -n pydl python3 src/train.py

# 3. Export to ONNX + INT8 quantization
conda run -n pydl python3 src/quantize.py

# 4. Evaluate on test set
conda run -n pydl python3 src/evaluate.py

# 5. Benchmark latency (run on Pi for real numbers)
conda run -n pydl python3 src/benchmark.py
```

### Raspberry Pi Deployment

```bash
# Clone and set up environment
git clone https://github.com/vanshdhar999/EdgeAI-Project.git
cd EdgeAI-Project
bash deployment/setup_pi.sh

# Run live inference
export DISPLAY=:0
source venv/bin/activate
python3 deployment/live_camera.py
```

Press **Q** to quit.

**Behaviour:** 10 s warmup on start → scanning (grey overlay) → detection locks for 10 s when confidence ≥ 80% → reverts to scanning.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [PlantVillage — Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| License | CC BY 4.0 |
| Classes | 15 (Tomato×10, Potato×3, Pepper×2) |
| Images/class | Capped at 1,000 |
| Split | 70% train / 15% val / 15% test |

Raw dataset excluded from git — download separately.

---

## Results

| Metric | float32 | INT8 |
|--------|---------|------|
| Model size | 3.8 MB | **1.1 MB** |
| Pi 5 latency (avg) | 17.3 ms | **6.1 ms** |
| Target latency | 1,500 ms | 1,500 ms |
| Accuracy drop | — | < 2% |

See [`docs/report.md`](docs/report.md) for full results, confusion matrix, and architecture details.

---
## One Drive Link For Demo
https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/vanshdhar_iisc_ac_in/IQAHf-pcBJ_YR5uXNs8KaSeoATctxBU7P9jHByqUmpHVHFg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=L5qe0P

---
## Team

| Name | Roll No. | Contribution |
|------|----------|-------------|
| Aayush Jeevan Patil | 22220 | Dataset pipeline, augmentation, model training, evaluation |
| Vansh Dhar | 22156 | ONNX export, INT8 quantization, Pi deployment, live camera system |

---

## Disclaimer

This project was developed with assistance from [Claude](https://claude.ai) (Anthropic), an AI assistant. Claude was used to help with code development, debugging, documentation, and architectural decisions throughout the project. All work has been reviewed, understood, and validated by the team members. This disclaimer is included to maintain academic integrity and transparency.
