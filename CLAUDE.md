# CLAUDE.md — Edge AI Plant Disease Detection System
> **Course:** Edge AI  
> **Start Date:** April 16, 2026  
> **Timeline:** ~1 week  
> **Primary Hardware:** Raspberry Pi + Raspberry Pi Camera Module v2  
> **AI Assistant:** Claude Code (claude-sonnet-4-20250514 recommended)  
> **GitHub Repo:** `https://github.com/vanshdhar999/EdgeAI-Project` ← update this

---

## Project Overview

Farmers in rural and semi-urban areas lack timely access to agricultural experts, causing undetected crop diseases and significant yield losses. This project builds an **offline, edge-deployable plant disease detection system** that runs directly on a Raspberry Pi. Using computer vision and a lightweight deep learning model trained on the PlantVillage dataset, the system classifies plant leaves as healthy or diseased (with specific disease type) in real time — no internet connectivity required.

**The end user:** A farmer in the field with no technical background, pointing a camera at a leaf and receiving an instant diagnosis.

---

## Realistic Deliverable (Definition of Done)

- [ ] Raspberry Pi + Camera Module displays a **live video feed**
- [ ] When a leaf is held up, the system overlays the **predicted disease class + confidence score** on the frame
- [ ] Inference completes within **1–2 seconds per frame**, entirely offline
- [ ] Tested on **3–4 disease categories** using physical leaf samples or printed reference images
- [ ] Model runs as a **TensorFlow Lite (.tflite)** file with INT8 quantization

---

## Hardware & Environment

| Component | Details |
|-----------|---------|
| Board | Raspberry Pi 5 Model B Rev 1.1 (confirmed via /proc/cpuinfo) |
| RAM | 7.9 GB total, ~6.7 GB available at idle |
| Camera | Connected via CSI (rp1-cfe, 900 Mbps link rate confirmed in dmesg) |
| OS | Debian GNU/Linux 12 (Bookworm), aarch64, kernel 6.12.75+rpt-rpi-2712 |
| Runtime | Python 3.11.2 |
| Disk | 58 GB total, ~46 GB free |
| CPU Temp (idle) | 69.2°C — warm at idle; monitor under sustained inference load |
| Connectivity | None required at inference time (fully offline) |
| Development Machine | Your laptop/desktop for training (GPU preferred) |
| Pre-installed packages | numpy 1.24.2, picamera2 0.3.31 |
| Missing packages | tflite-runtime, opencv-python (install via setup_pi.sh) |

**Key constraint:** The Pi's CPU is the inference engine. No GPU on-device. Every millisecond of model latency matters.

---

## Tech Stack

| Layer | Tool / Library | Reason |
|-------|---------------|--------|
| Dataset | PlantVillage (open-source, Kaggle) | Standard benchmark for plant disease |
| Training Framework | TensorFlow / Keras | Best TFLite export path |
| Base Model | MobileNetV3-Small or EfficientNet-Lite0 | Lightweight, proven on edge hardware |
| Quantization | TF Post-Training Quantization (INT8) | Reduces model size ~4x, speeds up inference |
| Edge Runtime | TensorFlow Lite (tflite-runtime) | Optimized for ARM CPUs |
| Camera Interface | OpenCV + picamera2 | Live feed capture on Pi |
| Visualization | OpenCV (cv2.putText, cv2.imshow) | Overlay labels on frames |
| Augmentation | TensorFlow `tf.image` or Albumentations | Field-condition robustness |

---

## Project Structure

```
plant-disease-edge/
│
├── CLAUDE.md                   # This file — project context for Claude Code
├── README.md                   # Human-readable project summary
│
├── data/
│   ├── raw/                    # Original PlantVillage dataset (do not modify)
│   ├── processed/              # Resized, split (train/val/test) images
│   └── augmented/              # Augmented training samples (optional)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_quantization_eval.ipynb
│
├── src/
│   ├── data_prep.py            # Download, split, preprocess PlantVillage
│   ├── augmentation.py         # Augmentation pipeline definitions
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Accuracy, confusion matrix, per-class metrics
│   ├── quantize.py             # INT8 post-training quantization + TFLite export
│   └── benchmark.py            # Latency benchmarking on Pi or desktop
│
├── models/
│   ├── checkpoints/            # .h5 / SavedModel during training
│   ├── plant_disease.tflite    # Final quantized model for deployment
│   └── labels.txt              # Class label list (one per line)
│
├── deployment/
│   ├── inference.py            # Single-image inference using TFLite runtime
│   ├── live_camera.py          # Main script: live feed + overlay on Pi
│   ├── requirements_pi.txt     # Minimal Pi dependencies
│   └── setup_pi.sh             # Pi environment setup script
│
├── tests/
│   ├── test_inference.py       # Unit test: model loads and returns valid output
│   └── test_pipeline.py        # Integration test: image → label → confidence
│
└── docs/
    ├── architecture.md         # Model architecture decisions and tradeoffs
    ├── dataset_notes.md        # Dataset stats, class distribution, cleaning steps
    └── results.md              # Accuracy, latency, real-leaf test results
```

---

## Development Phases & Tasks

### Phase 1 — Data Preparation (Day 1–2)
**Goal:** Have a clean, augmented dataset ready for training.

- [ ] Download PlantVillage dataset from Kaggle or TensorFlow Datasets
- [ ] Audit class distribution — identify imbalanced classes
- [ ] Resize all images to **224×224** (or 96×96 for faster iteration)
- [ ] Split into train (70%) / val (15%) / test (15%) — stratified by class
- [ ] Implement augmentation pipeline:
  - Random horizontal/vertical flip
  - Random rotation (±30°)
  - Brightness & contrast jitter
  - Optional: random zoom, Gaussian noise
- [ ] Verify no data leakage between splits
- [ ] Save `labels.txt` with class index → disease name mapping

**Key decision:** Start with a **subset** (e.g., 5–10 disease classes) to validate the pipeline quickly before scaling to the full dataset.

#### ✅ Phase 1 Git Checkpoint
Once all Phase 1 tasks are complete, Claude Code should run:
```bash
git add data/processed/ src/data_prep.py src/augmentation.py models/labels.txt docs/dataset_notes.md
git commit -m "feat(data): complete Phase 1 data preparation pipeline

- Download and preprocess PlantVillage dataset
- Implement stratified train/val/test split (70/15/15)
- Add augmentation pipeline: flip, rotation, brightness/contrast jitter
- Resize images to 224x224 and normalize
- Generate labels.txt with class index mappings
- Verify no data leakage across splits"
git push origin main
```

---

### Phase 2 — Model Training (Day 2–3)
**Goal:** A trained model with >85% validation accuracy on the chosen classes.

- [ ] Load pretrained MobileNetV3-Small (ImageNet weights) from `tf.keras.applications`
- [ ] Replace top classification head with:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dropout(0.3)`
  - `Dense(num_classes, activation='softmax')`
- [ ] Training strategy:
  - **Stage 1:** Freeze base, train head only (5–10 epochs, lr=1e-3)
  - **Stage 2:** Unfreeze top 20–30 layers, fine-tune (10–15 epochs, lr=1e-5)
- [ ] Use `tf.data` pipeline with prefetching and caching for speed
- [ ] Log metrics: accuracy, val_accuracy, loss, val_loss
- [ ] Save best checkpoint via `ModelCheckpoint` callback
- [ ] Generate confusion matrix and per-class precision/recall on test set

**Fallback:** If MobileNetV3 underperforms, try EfficientNet-Lite0 — same TFLite compatibility.

#### ✅ Phase 2 Git Checkpoint
Once all Phase 2 tasks are complete, Claude Code should run:
```bash
git add src/train.py src/evaluate.py notebooks/02_model_training.ipynb docs/architecture.md
git commit -m "feat(model): complete Phase 2 model training

- Fine-tune MobileNetV3-Small on PlantVillage subset
- Two-stage training: frozen base then partial unfreeze
- Achieved val_accuracy: <INSERT_VALUE>% on <N> disease classes
- Saved best checkpoint to models/checkpoints/
- Generated confusion matrix and per-class precision/recall"
git push origin main
```
> 📝 Fill in actual accuracy and class count before committing.

---

### Phase 3 — Model Compression & TFLite Export (Day 3–4)
**Goal:** A `.tflite` model under 10MB that runs in <1s on the Pi.

- [ ] Convert SavedModel → TFLite (float32 baseline)
- [ ] Apply **INT8 post-training quantization** using a representative dataset (100–200 samples from training set)
- [ ] Compare:
  - Model size: float32 vs INT8
  - Accuracy drop: should be <1–2%
  - Simulated latency (benchmark on desktop first)
- [ ] Validate TFLite model output matches Keras model output on same inputs
- [ ] Save final `plant_disease.tflite` + `labels.txt` to `models/`

**Explore if time permits:**
- Pruning (weight sparsity) before quantization
- Weight clustering

#### ✅ Phase 3 Git Checkpoint
Once all Phase 3 tasks are complete, Claude Code should run:
```bash
git add src/quantize.py src/benchmark.py models/plant_disease.tflite models/labels.txt notebooks/03_quantization_eval.ipynb
git commit -m "feat(compression): complete Phase 3 INT8 quantization and TFLite export

- Convert SavedModel to TFLite (float32 baseline)
- Apply INT8 post-training quantization with representative dataset
- Model size: float32=<X>MB → INT8=<Y>MB (<Z>x reduction)
- Accuracy drop: <A>% (float32) → <B>% (INT8)
- Validated TFLite output matches Keras model on test samples"
git push origin main
```
> 📝 Fill in actual file sizes and accuracy figures before committing.

---

### Phase 4 — Pi Deployment & Live Inference (Day 4–6)
**Goal:** Live camera feed with overlaid disease classification on the Raspberry Pi.

- [ ] Set up Pi environment (`setup_pi.sh`):
  ```bash
  sudo apt update
  pip install tflite-runtime opencv-python picamera2
  ```
- [ ] Write `inference.py`: load `.tflite`, preprocess input, run inference, return top-1 label + confidence
- [ ] Write `live_camera.py`:
  - Capture frames with `picamera2` or `cv2.VideoCapture`
  - Preprocess each frame (resize to 224×224, normalize to [0,1])
  - Run inference every N frames (tune for latency vs. responsiveness)
  - Overlay label + confidence score using `cv2.putText`
  - Display with `cv2.imshow` or pipe to HDMI display
- [ ] Measure actual inference latency on Pi (target: <1s per inference)
- [ ] Tune frame skip rate if needed (e.g., infer every 3rd frame)

#### ✅ Phase 4 Git Checkpoint
Once all Phase 4 tasks are complete, Claude Code should run:
```bash
git add deployment/ docs/architecture.md
git commit -m "feat(deploy): complete Phase 4 Pi deployment and live inference

- Implement TFLite inference pipeline (inference.py)
- Build live camera feed with disease overlay (live_camera.py)
- Measured inference latency on Pi: ~<X>ms per frame
- Frame skip rate set to every <N> frames for smooth display
- Confirmed fully offline operation on Raspberry Pi"
git push origin main
```
> 📝 Fill in measured latency and frame skip rate before committing.

---

### Phase 5 — Testing & Documentation (Day 6–7)
**Goal:** Demonstrate generalization and document everything clearly.

- [ ] Test with **physical leaf samples** OR printed high-quality reference images
- [ ] Cover at least **3–4 disease categories** from the trained classes
- [ ] Record results in `docs/results.md`:
  - Class tested, prediction, confidence, correct/incorrect
- [ ] Write `README.md` with:
  - Project summary
  - Setup instructions (dev machine + Pi)
  - How to run training and deployment
  - Sample output screenshot/video
- [ ] Complete `docs/architecture.md` with model choice rationale

#### ✅ Phase 5 Git Checkpoint — Final Release
Once all Phase 5 tasks are complete, Claude Code should run:
```bash
git add README.md docs/results.md docs/architecture.md tests/
git commit -m "feat(final): complete Phase 5 testing and documentation

- Tested on <N> real leaf samples across <M> disease categories
- Pass rate: <X>/<N> correct predictions
- Completed README with setup instructions and usage guide
- Documented model architecture decisions and tradeoffs
- Recorded full results in docs/results.md"

git tag -a v1.0.0 -m "Edge AI Plant Disease Detection — Course Project Complete"
git push origin main
git push origin v1.0.0
```

---

## Key Technical Constraints & Decisions

### Why MobileNetV3-Small over larger models?
- Designed for mobile/edge CPUs — low FLOPs, low memory footprint
- ImageNet pretrained weights available in Keras — transfer learning is fast
- Excellent TFLite support with quantization-aware design

### Why INT8 quantization?
- Reduces model size by ~4x (float32 → int8)
- ARM Cortex-A CPUs (Pi) have optimized INT8 SIMD instructions
- Typical accuracy drop: <1–2% on well-trained models

### Input resolution tradeoff
- 224×224: Standard MobileNet input, higher accuracy
- 96×96 or 128×128: Faster preprocessing + inference, slightly lower accuracy
- **Recommendation:** Train at 224×224, benchmark on Pi, reduce if latency is too high

### Camera pipeline
- Use `picamera2` (modern Pi camera API, supports Pi OS Bullseye+)
- Fall back to `cv2.VideoCapture(0)` if picamera2 has issues
- Frame capture is separate from inference — don't block capture waiting for model

---

## Git Workflow

The project uses a single `main` branch. Claude Code must follow this workflow throughout:

**Repository setup (run once at project start):**
```bash
git init
git remote add origin https://github.com/vanshdhar999/EdgeAI-Project.git
git add CLAUDE.md README.md .gitignore
git commit -m "chore: initialize project with CLAUDE.md and structure"
git push -u origin main
```

**`.gitignore` must include:**
```
data/raw/
data/augmented/
models/checkpoints/
*.h5
__pycache__/
.env
*.pyc
```

**Commit message format** (Conventional Commits):
```
<type>(<scope>): <short summary>

<optional body with bullet points>
```
Types: `feat`, `fix`, `chore`, `docs`, `refactor`, `test`  
Scopes: `data`, `model`, `compression`, `deploy`, `docs`, `ci`

**Rules for Claude Code:**
- Never `git push --force` on `main`
- Never commit large binary files (datasets, `.h5` checkpoints, large `.tflite` > 25MB)
- Always verify `git status` is clean before pushing
- If a push is blocked, diagnose and report — never bypass protections silently

---

## Raspberry Pi — Hardware Profiling and Constraints

This will only work when my mac is connected to my phone's hotspot. So before executing this please alert me to switch my wifi network. 

### Connecting to the Pi
```bash
ssh rpi9@172.20.10.2
password = 12345
```
> If the Pi's IP is unknown, run `hostname -I` on the Pi directly, or check your router's connected devices list.

### Hardware Profiling Commands
Once connected, Claude Code should run the following and record the outputs in the Hardware & Environment table above:

```bash
# Pi model and CPU
cat /proc/cpuinfo | grep -E "Model|Hardware|Revision" | head -5

# RAM
free -h

# OS and architecture
uname -a
cat /etc/os-release | grep -E "PRETTY_NAME|VERSION"

# Python version
python3 --version

# Disk space available
df -h /

# Check if camera is detected
libcamera-hello --list-cameras 2>/dev/null || vcgencmd get_camera

# CPU temperature (thermal constraint awareness)
vcgencmd measure_temp

# Check for existing Python packages relevant to this project
pip3 list 2>/dev/null | grep -iE "tflite|tensorflow|opencv|picamera|numpy"
```

### What to do with the results
After running the above, Claude Code should:
1. Update the **Hardware & Environment** table in this file with actual values (Pi model, RAM, OS version, Python version)
2. Adjust latency targets if the Pi is a lower-spec model (e.g., Pi 3B has less RAM and a slower CPU than Pi 4)
3. Flag any missing packages that need to be installed via `setup_pi.sh`
4. Note the thermal headroom — if the Pi runs hot, sustained inference may throttle

### Getting Code onto the Pi
The Pi will run directly from a clone of the GitHub repo. No manual file transfers needed.

```bash
# On the Pi (via SSH), clone the repo once
git clone https://github.com/vanshdhar999/EdgeAI-Project.git
cd plant-disease-edge

# For subsequent updates, just pull the latest changes
git pull origin main
```

This means every Phase git push on the dev machine is immediately available on the Pi with a simple `git pull`. Make sure the `.tflite` model is committed (if under 25MB) or document a download step in `setup_pi.sh` if it exceeds GitHub's file size limits.

---



> These notes help Claude Code understand how to assist effectively in this project.

- **Language:** Python 3.9+. Prefer explicit, readable code over one-liners.
- **Framework:** TensorFlow/Keras for training; `tflite-runtime` (not full TF) on the Pi.
- **No GPU on Pi** — never suggest CUDA/GPU code for deployment scripts.
- **File paths:** Use `pathlib.Path` throughout, not raw strings.
- **Model files:** Never commit large model files to git. Add `models/checkpoints/` and `data/raw/` to `.gitignore`.
- **Testing:** Write at minimum a smoke test for inference — load model, pass a dummy tensor, check output shape.
- **When suggesting model changes:** Always consider the Pi's constraints. If a model layer isn't TFLite-compatible, flag it immediately.
- **Preferred style:** PEP8, docstrings on all functions, type hints where practical.
- **When asked to debug inference issues on Pi:** First check if the error is in the TFLite runtime vs. the camera pipeline vs. preprocessing — they fail differently.

---

## Common Pitfalls to Avoid

| Pitfall | Mitigation |
|--------|-----------|
| Data leakage between train/test splits | Use stratified split, verify patient/image IDs don't cross splits |
| Overfitting on PlantVillage (lab images) | Heavy augmentation; test on real field photos |
| TFLite incompatible layers | Stick to standard Keras layers; avoid custom ops |
| Wrong input normalization at inference | Match exactly: `pixel / 255.0` or `(pixel - mean) / std` — must be identical in training and deployment |
| Pi running out of memory | Use `tflite-runtime` only (not full TF); reduce batch size |
| Slow frame rate from blocking inference | Run inference in a separate thread or skip frames |
| Labels mismatch between model and labels.txt | Auto-generate labels.txt from `class_names` during training |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Validation accuracy | ≥ 85% on chosen disease classes |
| INT8 accuracy drop | < 2% vs. float32 baseline |
| Inference latency on Pi | < 1.5 seconds per frame |
| Model file size | < 10 MB (.tflite, INT8) |
| Real-leaf test pass rate | Correct prediction on ≥ 3 of 4 disease categories |

---

## References & Resources

- **PlantVillage Dataset:** https://www.kaggle.com/datasets/emmarex/plantdisease
- **TensorFlow Lite Guide:** https://www.tensorflow.org/lite/guide
- **MobileNetV3 Paper:** Howard et al., 2019 — "Searching for MobileNetV3"
- **TFLite Post-Training Quantization:** https://www.tensorflow.org/lite/performance/post_training_quantization
- **picamera2 Docs:** https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
- **Claude Code Docs:** https://docs.claude.com/en/docs/claude-code/overview

---

*This CLAUDE.md is a living document. Update it as decisions are made, constraints change, or new findings emerge during the project.*