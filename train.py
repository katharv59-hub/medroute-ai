"""
ASEP — YOLOv8 Training Script
================================
Trains ambulance detection model optimized for RTX 4050.
"""
from ultralytics import YOLO
import os, sys
from config import (
    BASE_MODEL, DATASET_YAML, TRAIN_EPOCHS, TRAIN_BATCH,
    TRAIN_PATIENCE, TRAIN_WORKERS, TRAIN_AMP, FRAME_RESIZE, AUGMENTATION
)

def train():
    if not os.path.exists(DATASET_YAML):
        print(f"[ERROR] Dataset not found: {DATASET_YAML}")
        sys.exit(1)

    model = YOLO(BASE_MODEL)
    print(f"[TRAIN] Base: {BASE_MODEL} | Epochs: {TRAIN_EPOCHS} | Batch: {TRAIN_BATCH}")

    model.train(
        data=DATASET_YAML, epochs=TRAIN_EPOCHS, imgsz=FRAME_RESIZE,
        batch=TRAIN_BATCH, name="ambulance_model_v2", project="runs/detect",
        patience=TRAIN_PATIENCE, device=0, amp=TRAIN_AMP,
        workers=TRAIN_WORKERS, cache=True, **AUGMENTATION,
    )

    print("\n[TRAIN] Complete! Running validation...")
    best = YOLO("runs/detect/ambulance_model_v2/weights/best.pt")
    m = best.val(data=DATASET_YAML, device=0)
    print(f"\n  mAP50: {m.box.map50:.4f} | mAP50-95: {m.box.map:.4f}")
    print(f"  Precision: {m.box.mp:.4f} | Recall: {m.box.mr:.4f}")
    print("  ✓ PASS" if m.box.map50 >= 0.90 else "  ✗ Need more data/epochs")

if __name__ == '__main__':
    print("=" * 50)
    print("  ASEP — Model Training (RTX 4050 + CUDA 12.1)")
    print("=" * 50)
    train()
