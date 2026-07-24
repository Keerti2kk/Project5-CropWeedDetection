# 🌿 Crop and Weed Detection using YOLOv5

## 📋 Project Overview
A deep learning **object detection system** using YOLOv5 that detects and localizes **sesame crops** and **weed species** in field images. Enables precision pesticide spraying — targeting only weeds, not crops.

**Company:** UCT (Universal Computer Technologies)  
**Domain:** Computer Vision / Deep Learning  
**Internship:** Week 1 - Week 4  

---

## 🎯 Problem Statement

Weeds are unwanted plants that steal nutrients, water, and land from crops, resulting in lower production.

**Challenges:**
- Farmers use blanket pesticide spraying across entire fields
- Chemicals stick to edible crops, causing health hazards
- Wasteful and environmentally harmful
- Reduces crop yield by 20-40%

**Solution:**
A computer vision system that identifies weeds at pixel level, enabling:
- ✅ Precision pesticide spraying (weeds only)
- ✅ Reduced chemical waste
- ✅ Improved food safety
- ✅ Lower operational costs

---

## 📊 Dataset

**Total Images:** 1,300 (512×512 color images)  
**Classes:** 2 — Sesame Crop [0], Weed [1]  
**Format:** YOLO annotation (.txt label files with normalized bounding boxes)  
**Download:** [Google Drive Link](https://drive.google.com/file/d/1MNdDKYB0x0PEW7P71bE1Jx_uLllvORA0/view?usp=sharing)

### Dataset Preparation Pipeline
```
589 raw field images
  → cleaned (removed blurry/bad photos) → 546 images
  → resized to 512×512 pixels
  → augmented (flip, rotate, zoom, brightness) → 1,300 images
  → manually labeled with bounding boxes (LabelImg tool)
  → split into train (80%) / val (10%) / test (10%)
```

---

## 📁 Project Structure
```
Project5/
│
├── README.md                    ← You are here
├── .gitignore
├── requirements.txt
├── dataset.yaml                 ← YOLO dataset configuration
│
├── src/
│   ├── data_augmentation.py    ← Augment images 546→1300
│   ├── train.py                ← YOLOv5 training script
│   └── inference.py            ← Run detection on new images
│
├── data/
│   └── README.md               ← Dataset download instructions
│
├── results/
│   ├── detections/             ← Annotated detection outputs
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   └── evaluation_metrics.txt
│
├── weights/
│   └── best.pt                 ← Trained model (not in GitHub)
│
└── notebooks/
    └── training_log.ipynb
```

---

## 🧰 Libraries Used
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python==4.8.0.76
Pillow==10.0.0
matplotlib==3.7.2
PyYAML==6.0.1
tqdm==4.66.1
numpy==1.24.3
seaborn==0.12.2
onnxruntime==1.16.0
```

---

## ▶️ How to Run

### Step 1: Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/Project5-CropWeedDetection
cd Project5-CropWeedDetection

pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5.git
cd yolov5 && pip install -r requirements.txt && cd ..
```

### Step 2: Prepare Dataset
```bash
# Download from Google Drive link above
# Place in data/ folder and organize:

data/
├── images/
│   ├── train/   (1040 images - 80%)
│   ├── val/     (130 images - 10%)
│   └── test/    (130 images - 10%)
└── labels/
    ├── train/   (1040 .txt files)
    ├── val/     (130 .txt files)
    └── test/    (130 .txt files)
```

### Step 3: Data Augmentation (Optional)
```bash
python src/data_augmentation.py
```
Expands 546 clean images → 1,300 augmented images with bounding boxes.

### Step 4: Train the Model
```bash
python src/train.py
```
Trains YOLOv5s for 50 epochs with:
- Batch size: 16
- Learning rate: 0.01
- Optimizer: SGD with momentum
- Pretrained COCO weights

### Step 5: Run Inference
```bash
python src/inference.py --input_folder ./test_images --output_folder ./results/detections
```
Runs detection on all images in input folder, outputs annotated images.

---

## 📈 Results & Performance

### Model Comparison
| Model | Precision | Recall | F1-Score | mAP@0.5 | Inference (ms/img) |
|---|---|---|---|---|---|
| YOLOv5s (Final) ⭐ | 0.87 | 0.84 | 0.85 | **0.85** | **47** |
| YOLOv5m | 0.89 | 0.86 | 0.87 | 0.89 | 82 |

### Class-wise Performance (YOLOv5s)
| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Sesame Crop | 0.89 | 0.87 | 0.88 |
| Weed | 0.85 | 0.81 | 0.83 |

### Key Findings
- ✅ **YOLOv5s selected as production model** — fastest inference for drone/robot deployment
- ✅ **Real-time capable** — <50ms per image on CPU
- ✅ **Strong weed detection** — 85% mAP across diverse weed species
- ✅ **ONNX compatible** — deployable on edge devices

### Output Charts
✅ `confusion_matrix.png` — Prediction accuracy per class  
✅ `pr_curve.png` — Precision-Recall curves  
✅ `evaluation_metrics.txt` — Detailed metrics summary  

---

## 🔮 Inference Example

```python
from src.inference import run_detection

# Detect weeds and crops in an image
input_image = "field_photo.jpg"
output_image = "field_photo_annotated.jpg"

run_detection(
    image_path=input_image,
    model_path="weights/best.pt",
    conf_threshold=0.5,
    output_path=output_image
)

# Output: field_photo_annotated.jpg with bounding boxes & labels
```

---

## 🎓 Learnings & Insights

### Week 1 - Data Preparation
- Captured 589 real field images with smartphone camera
- Cleaned dataset: removed blurry/duplicate images → 546
- Learned data augmentation techniques (flip, rotate, zoom, brightness)
- Manual bounding box labeling with LabelImg tool
- Created 1,300 training images from 546 originals

### Week 2 - Model Training
- Trained YOLOv5s with transfer learning from COCO weights
- Ran full 50-epoch training with GPU acceleration
- Evaluated using mAP@0.5, mAP@0.5:0.95, precision, recall
- Analyzed training curves to diagnose convergence

### Week 3 - Model Selection & Export
- Compared YOLOv5s vs YOLOv5m for accuracy-speed tradeoff
- Selected YOLOv5s for edge device deployment
- Exported to ONNX for cross-platform compatibility
- Verified ONNX Runtime inference on test images

### Week 4 - Finalization & Deployment
- Built complete inference demo script
- Tested on Raspberry Pi (GPIO/camera integration ready)
- Prepared GitHub documentation & project submission
- Delivered functional weed detection system

---

## 🚀 Future Scope

1. **Edge Deployment**
   - Integrate with Raspberry Pi camera + GPIO controls
   - Deploy on agricultural drone (DJI SDK integration)
   - Test on autonomous tractor/robot platforms

2. **Model Improvements**
   - Increase input resolution to 640×640 for better small weed detection
   - Collect more weed species (currently: sesame + local weeds)
   - Train YOLOv8 for potential accuracy boost

3. **System Integration**
   - Link detection output to pesticide sprayer controls
   - Real-time field video feed processing
   - Farmer mobile app with cloud inference

4. **Scalability**
   - Multi-crop detection (beyond sesame)
   - Region-specific model training
   - Seasonal fine-tuning

---

## 👥 Author
**Keerti Chauhan**  
Computer Vision Intern — UCT (Universal Computer Technologies)  
Week 1–4, June–July 2026

---

## 📞 Support & Feedback
For questions, issues, or feedback:
- Open an issue on GitHub
- Contact: [your email]
- LinkedIn: [your profile]

---

## 📜 License
This project is licensed under the MIT License — see LICENSE file for details.

---

## 🙏 Acknowledgements
- **Ultralytics YOLOv5** — State-of-the-art object detection framework
- **Company:** UCT (Universal Computer Technologies)
- **Mentors & Peers:** UCT internship team for guidance and feedback
- **Dataset:** Field images captured with team collaboration
