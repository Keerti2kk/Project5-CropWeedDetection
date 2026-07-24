# Notebooks

## training_log.ipynb

Complete Jupyter notebook documenting the YOLOv5 training process for crop and weed detection.

### How to use:

**Option 1: Google Colab (Recommended for GPU)**
```
1. Open colab.research.google.com
2. Click "Upload" → select this notebook
3. Download dataset from Google Drive link in README
4. Upload to Colab when prompted
5. Run all cells with GPU enabled (Runtime → Change runtime type → GPU)
```

**Option 2: Local Jupyter**
```bash
pip install jupyter
jupyter notebook training_log.ipynb
```

---

## Contents

- 📦 Environment setup
- 📊 Dataset structure & statistics
- 📷 Sample image visualization
- 🎯 YOLO annotation inspection
- 🚀 YOLOv5 model initialization
- 📈 Training loop with progress tracking
- 📊 Evaluation metrics (mAP, precision, recall)
- 📉 Training curves & loss graphs
- 🔍 Inference on test images
- 💾 Model export (ONNX)

---

## Key Sections

### 1. Setup
```python
!pip install -q yolov5
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
```

### 2. Training
```python
!python train.py \
    --img 512 \
    --batch 16 \
    --epochs 50 \
    --data ../dataset.yaml \
    --weights yolov5s.pt \
    --device 0
```

### 3. Results Visualization
- Training/validation loss curves
- mAP@0.5 & mAP@0.5:0.95 progression
- Confusion matrix heatmap
- Precision-Recall curves

### 4. Inference Demo
```python
from yolov5 import YOLOv5
model = YOLOv5('runs/train/exp/weights/best.pt')
results = model.predict('test_image.jpg')
```

---

## Requirements

All dependencies listed in `requirements.txt` at project root.

```bash
pip install -r ../requirements.txt
```

Plus YOLOv5:
```bash
pip install yolov5
# or
git clone https://github.com/ultralytics/yolov5
cd yolov5 && pip install -r requirements.txt
```

---

## Expected Runtime

- **Google Colab (GPU):** ~2-3 hours for 50 epochs
- **Local PC (GPU):** ~3-5 hours for 50 epochs
- **CPU only:** Not recommended (very slow)

---

## Output

Training produces:
- `runs/train/exp/weights/best.pt` — Best model weights
- `runs/train/exp/weights/last.pt` — Last checkpoint
- `runs/train/exp/results.csv` — Metrics log
- `runs/train/exp/confusion_matrix.png` — Confusion matrix
- `runs/train/exp/pr_curve.png` — Precision-Recall curve

---

## Tips

✅ **Use Google Colab** for faster training with free GPU  
✅ **Monitor GPU usage** with `!nvidia-smi` cell  
✅ **Save weights to Google Drive** for persistence  
✅ **Review training curves** to spot overfitting  
✅ **Test on diverse images** before deployment  

---

## Troubleshooting

**CUDA memory error?**
→ Reduce batch size from 16 to 8 or 4

**Slow training?**
→ Use Google Colab GPU or local GPU instead of CPU

**Model not converging?**
→ Adjust learning rate or increase epochs

---

Questions? Check YOLOv5 official docs:  
https://github.com/ultralytics/yolov5
