"""
Project 5: Crop and Weed Detection
File: src/train.py
Description: Trains YOLOv5 by invoking the official Ultralytics train.py
Company: UCT — Machine Learning Internship

IMPORTANT: This script does NOT reimplement YOLO training itself.
Training a YOLO model from scratch is complex (loss functions, anchors,
NMS, augmentation pipeline, etc.) and Ultralytics' own train.py is the
tested, correct, actively-maintained way to do it. This script is a thin
wrapper that calls it correctly with this project's settings, so you don't
have to remember the exact command.

Prerequisite (one-time):
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
"""

import os
import subprocess
import sys

# ── TRAINING CONFIGURATION ──────────────────────────────────
CONFIG = {
    'yolov5_dir': '../yolov5',        # path to cloned yolov5 repo
    'data_yaml': '../dataset.yaml',   # relative to yolov5_dir
    'weights': 'yolov5s.pt',          # pretrained COCO weights (auto-downloaded)
    'img_size': 512,
    'batch_size': 16,
    'epochs': 50,
    'device': '0',                    # '0' for first GPU, 'cpu' for CPU
    'save_period': 5,                 # checkpoint every N epochs
    'project': '../results',          # where YOLOv5 saves run outputs
    'name': 'crop_weed_detection',
}


def train():
    yolov5_dir = CONFIG['yolov5_dir']

    if not os.path.isdir(yolov5_dir):
        print(f"❌ YOLOv5 repo not found at '{yolov5_dir}'")
        print("   Clone it first:")
        print("   git clone https://github.com/ultralytics/yolov5")
        sys.exit(1)

    train_script = os.path.join(yolov5_dir, 'train.py')
    if not os.path.exists(train_script):
        print(f"❌ train.py not found in '{yolov5_dir}'")
        sys.exit(1)

    cmd = [
        sys.executable, 'train.py',
        '--img', str(CONFIG['img_size']),
        '--batch', str(CONFIG['batch_size']),
        '--epochs', str(CONFIG['epochs']),
        '--data', CONFIG['data_yaml'],
        '--weights', CONFIG['weights'],
        '--device', CONFIG['device'],
        '--save-period', str(CONFIG['save_period']),
        '--project', CONFIG['project'],
        '--name', CONFIG['name'],
    ]

    print("=" * 60)
    print("CROP AND WEED DETECTION — YOLOv5 TRAINING")
    print("Company: UCT — Machine Learning Internship")
    print("=" * 60)
    print("\n🔧 Running command:")
    print("   " + " ".join(cmd))
    print(f"\n   (executed from directory: {yolov5_dir})\n")

    result = subprocess.run(cmd, cwd=yolov5_dir)

    if result.returncode == 0:
        print("\n✅ Training complete!")
        print(f"   Best weights: {CONFIG['project']}/{CONFIG['name']}/weights/best.pt")
    else:
        print(f"\n❌ Training exited with error code {result.returncode}")
        print("   Check the YOLOv5 output above for details.")

    return result.returncode


if __name__ == "__main__":
    train()
