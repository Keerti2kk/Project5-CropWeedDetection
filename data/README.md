# Dataset — Crop and Weed Detection

## Download Instructions

The dataset is **NOT included** in this repository due to size constraints. Download it separately from:

👉 **[Google Drive Link](https://drive.google.com/file/d/1MNdDKYB0x0PEW7P71bE1Jx_uLllvORA0/view?usp=sharing)**

### Steps:
1. Click the Google Drive link above
2. Click **Download** button (or right-click → Save)
3. Extract the ZIP file
4. Place images and labels in this `data/` folder according to the structure below

---

## Dataset Structure

After downloading and extracting, organize files as:

```
Project5/
└── data/
    ├── README.md
    ├── images/
    │   ├── train/      (1040 images)
    │   ├── val/        (130 images)
    │   └── test/       (130 images)
    └── labels/
        ├── train/      (1040 .txt files)
        ├── val/        (130 .txt files)
        └── test/       (130 .txt files)
```

---

## Dataset Information

**Total Images:** 1,300  
**Image Format:** PNG / JPG (512×512 color)  
**Annotation Format:** YOLO (.txt label files)  
**Classes:** 2 — Crop [0], Weed [1]  
**Source:** Field-captured images  
**License:** Open for research use  

---

## Label Format

Each image has a corresponding `.txt` label file with the same name:

```
image_001.jpg  →  image_001.txt
```

Format (normalized YOLO coordinates):
```
<class_id> <x_center> <y_center> <width> <height>
0 0.512 0.456 0.234 0.123
1 0.234 0.789 0.098 0.145
```

Where:
- `class_id`: 0 for Crop, 1 for Weed
- `x_center, y_center`: Center of bounding box (0-1 normalized)
- `width, height`: Size of bounding box (0-1 normalized)

---

## Data Quality

- **1,300 manually annotated images**
- **Cleaned from 546 original images** via filtering
- **Augmented** using rotation, flip, zoom, brightness adjustments
- **Validated** for annotation accuracy (peer review)

---

## Usage

Once organized in the structure above, run:

```bash
python ../src/data_augmentation.py    # Optional: further augment
python ../src/train.py                 # Train the model
python ../src/inference.py --input test_image.jpg  # Run detection
```

---

## Statistics

| Set | Images | Labels |
|---|---|---|
| Training | 1,040 | 1,040 |
| Validation | 130 | 130 |
| Test | 130 | 130 |
| **Total** | **1,300** | **1,300** |

---

## Class Distribution

| Class | Count | Percentage |
|---|---|---|
| Crop | 1,820 | 45% |
| Weed | 2,240 | 55% |
| **Total** | **4,060** | **100%** |

---

## Notes

- Images may have multiple instances of crops and weeds
- Small weed seedlings (~50-100 pixels) are included
- Dataset represents typical sesame field conditions
- Captured under natural lighting (morning/afternoon)
