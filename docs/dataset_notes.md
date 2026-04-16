# Dataset Notes — PlantVillage Subset

## Source
- **Dataset:** PlantVillage (color variant)
- **Origin:** Kaggle — `emmarex/plantdisease`
- **Full dataset:** 38 classes, ~54,000 images
- **Local path:** `plantvillage-dataset/color/` (excluded from git via `.gitignore`)

---

## Subset Selection

For pipeline validation we selected **5 tomato disease classes**. Rationale:
- Single plant species reduces inter-class visual noise (background, leaf shape)
- All 5 classes have 1,000–2,127 images — no severe imbalance within the subset
- Covers the most agriculturally relevant tomato diseases
- Easily expandable to other plant families in a later iteration

| Index | Class | Raw Image Count |
|-------|-------|-----------------|
| 0 | Tomato___healthy | 1,591 |
| 1 | Tomato___Early_blight | 1,000 |
| 2 | Tomato___Late_blight | 1,909 |
| 3 | Tomato___Bacterial_spot | 2,127 |
| 4 | Tomato___Septoria_leaf_spot | 1,771 |
| — | **TOTAL** | **8,398** |

---

## Preprocessing

- **Input format:** Original JPEGs from PlantVillage (various resolutions, mostly 256×256)
- **Resize:** Lanczos resampling → 224×224 (standard MobileNetV3 input)
- **Color space:** Converted to RGB (drops any alpha channel, handles grayscale edge cases)
- **Output format:** JPEG, quality=95

---

## Train / Val / Test Split

| Split | Images | % |
|-------|--------|---|
| Train | 5,876 | 70.0% |
| Val | 1,258 | 15.0% |
| Test | 1,264 | 15.0% |
| **Total** | **8,398** | **100%** |

Split is **stratified per class** using a deterministic shuffle (`random_seed=42`).

### Per-class breakdown

| Class | Train | Val | Test |
|-------|-------|-----|------|
| Tomato___healthy | 1,113 | 238 | 240 |
| Tomato___Early_blight | 700 | 150 | 150 |
| Tomato___Late_blight | 1,336 | 286 | 287 |
| Tomato___Bacterial_spot | 1,488 | 319 | 320 |
| Tomato___Septoria_leaf_spot | 1,239 | 265 | 267 |

---

## Data Leakage Verification

Verified at prep time that no image filename (stem) appears in more than one split.
All three pairwise intersection checks (train∩val, train∩test, val∩test) returned empty sets.

---

## Known Limitations

- **Lab images:** PlantVillage images are taken under controlled lighting on plain backgrounds. Real-world field images have cluttered backgrounds, variable lighting, and partial leaf occlusion — expect an accuracy gap. Mitigated by augmentation (brightness, contrast, rotation, zoom).
- **Class imbalance in full dataset:** The full 38-class dataset ranges from 152 (`Potato___healthy`) to 5,507 (`Orange___Haunglongbing`) images. Addressed by restricting subset to balanced classes.
- **Single crop per leaf:** Most PlantVillage images show one leaf segment per file — no patient/plant ID to track, so standard filename-based leakage check is sufficient.

---

## Augmentation Pipeline

Defined in `src/augmentation.py`. Applied only to the **training split** at load time (not baked into files on disk):

| Transform | Parameters |
|-----------|------------|
| Random horizontal flip | p=0.5 |
| Random vertical flip | p=0.5 |
| Random rotation | ±30° (reflect padding) |
| Random brightness | delta=0.2 |
| Random contrast | scale ∈ [0.7, 1.3] |
| Random zoom crop | factor=0.1 (90–100% of image) |
| Gaussian noise | σ=0.02 (optional, off by default) |

---

## Metadata File

Full split metadata (image size, ratios, seed, per-class counts) saved to:
`data/processed/dataset_meta.json`
