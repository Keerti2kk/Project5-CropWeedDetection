"""
Project 5: Crop and Weed Detection
File: src/inference.py
Description: Run detection on new field images using the trained YOLOv5 model
Company: UCT — Machine Learning Internship

NOTE: This uses torch.hub.load('ultralytics/yolov5', 'custom', ...), which is
YOLOv5's official, documented, version-stable loading method. This is the
recommended approach over importing YOLOv5's internal functions directly,
which change between releases and can break silently.
"""

import os
import argparse
import glob

print("=" * 60)
print("CROP AND WEED DETECTION — INFERENCE DEMO")
print("Company: UCT — Machine Learning Internship")
print("=" * 60)


def load_model(model_path="weights/best.pt"):
    """
    Load the trained YOLOv5 model using torch.hub — YOLOv5's official,
    stable loading method (works across YOLOv5 versions).

    Requires internet on first run (downloads YOLOv5 repo code via hub),
    or a local yolov5/ folder if working fully offline.
    """
    try:
        import torch
    except ImportError:
        print("❌ PyTorch not installed. Install with: pip install torch")
        return None

    if not os.path.exists(model_path):
        print(f"❌ Model weights not found at: {model_path}")
        print("   Train a model first, or place your best.pt file there.")
        return None

    print(f"🔧 Loading model from: {model_path}")

    try:
        # Official YOLOv5 loading method — pulls architecture code from
        # the ultralytics/yolov5 GitHub repo (cached after first run)
        model = torch.hub.load('ultralytics/yolov5', 'custom',
                               path=model_path, force_reload=False)
    except Exception as e:
        print(f"❌ torch.hub load failed: {e}")
        print("   If offline, clone yolov5 locally and load with:")
        print("   torch.hub.load('./yolov5', 'custom', path=model_path, source='local')")
        return None

    print("✅ Model loaded successfully")
    return model


def run_detection(image_path, model, conf_threshold=0.5, output_path=None):
    """
    Run detection on a single image and optionally save the annotated result.

    Parameters:
        image_path     : str   — Path to input image
        model           : YOLOv5 model object (from load_model)
        conf_threshold  : float — Confidence threshold for detections
        output_path     : str  — Path to save annotated output image (optional)

    Returns:
        results object (pandas-convertible via .pandas().xyxy[0])
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    model.conf = conf_threshold  # set confidence threshold

    print(f"🔍 Running detection on: {image_path}")
    results = model(image_path)

    # Print detected objects as a table
    detections = results.pandas().xyxy[0]
    print(f"   ✅ Found {len(detections)} objects")
    if len(detections) > 0:
        print(detections[['name', 'confidence']].to_string(index=False))

    if output_path:
        save_dir = os.path.dirname(output_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        results.save(save_dir=save_dir)
        print(f"💾 Annotated image saved to: {save_dir}")

    return results


def run_batch_detection(input_folder, model, conf_threshold=0.5, output_folder="results/detections"):
    """Run detection on every image in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    image_paths = glob.glob(os.path.join(input_folder, "*.jpg")) + \
                  glob.glob(os.path.join(input_folder, "*.png"))

    print(f"\n📁 Found {len(image_paths)} images in '{input_folder}'")

    model.conf = conf_threshold
    all_results = []

    for img_path in image_paths:
        print(f"\n🔍 Processing: {os.path.basename(img_path)}")
        results = model(img_path)
        detections = results.pandas().xyxy[0]
        print(f"   Found {len(detections)} objects")
        results.save(save_dir=output_folder)
        all_results.append((img_path, len(detections)))

    print(f"\n✅ Batch detection complete! Results saved to: {output_folder}")
    return all_results


# ── COMMAND LINE INTERFACE ──────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop and Weed Detection Inference")
    parser.add_argument('--input', type=str, default=None,
                       help='Path to a single input image')
    parser.add_argument('--input_folder', type=str, default=None,
                       help='Path to a folder of images for batch detection')
    parser.add_argument('--model', type=str, default="weights/best.pt",
                       help='Path to trained YOLOv5 .pt model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default="results/detections",
                       help='Output folder for annotated images')

    args = parser.parse_args()

    model = load_model(args.model)

    if model is None:
        print("\n⚠️  Cannot run inference without a loaded model.")
        print("    Example usage:")
        print("    python inference.py --input field.jpg --model weights/best.pt")
        print("    python inference.py --input_folder ./test_images --model weights/best.pt")
        exit(1)

    if args.input:
        run_detection(args.input, model, args.conf, args.output)
    elif args.input_folder:
        run_batch_detection(args.input_folder, model, args.conf, args.output)
    else:
        print("\n⚠️  No input specified. Use --input <image> or --input_folder <folder>")

    print("\n✅ Detection Complete!")
