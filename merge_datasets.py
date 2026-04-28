"""
merge_datasets.py — Merge two YOLOv8 datasets into one unified dataset.

Source 1: dataset1/
Source 2: dataset2/
Output:   dataset_merged/

All label files are normalized to class ID 0 (ambulance).
"""

import os
import shutil


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET1 = os.path.join(BASE_DIR, "dataset1")
DATASET2 = os.path.join(BASE_DIR, "dataset2")
MERGED   = os.path.join(BASE_DIR, "dataset_merged")

SPLITS = ["train", "valid"]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def ensure_dirs(base, splits):
    """Create the merged directory tree."""
    for split in splits:
        os.makedirs(os.path.join(base, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(base, split, "labels"), exist_ok=True)


def copy_files(src_dir, dst_dir, prefix):
    """Copy every file from src_dir → dst_dir, adding *prefix* to filenames.

    Returns the number of files copied.
    """
    if not os.path.isdir(src_dir):
        return 0

    count = 0
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        if not os.path.isfile(src_path):
            continue
        dst_path = os.path.join(dst_dir, prefix + filename)
        shutil.copy2(src_path, dst_path)
        count += 1
    return count


def fix_labels(labels_dir):
    """Rewrite every .txt label file so every class ID becomes 0."""
    if not os.path.isdir(labels_dir):
        return

    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(labels_dir, filename)
        fixed_lines = []

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Replace class ID (first element) with 0
                    parts[0] = "0"
                    fixed_lines.append(" ".join(parts))

        with open(filepath, "w") as f:
            f.write("\n".join(fixed_lines))
            if fixed_lines:
                f.write("\n")


def create_data_yaml(merged_dir):
    """Write data.yaml for YOLOv8 training."""
    yaml_path = os.path.join(merged_dir, "data.yaml")
    train_path = os.path.join(merged_dir, "train", "images").replace("\\", "/")
    val_path   = os.path.join(merged_dir, "valid", "images").replace("\\", "/")

    content = (
        f"train: {train_path}\n"
        f"val: {val_path}\n"
        f"nc: 1\n"
        f"names: ['ambulance']\n"
    )

    with open(yaml_path, "w") as f:
        f.write(content)

    print(f"[✓] data.yaml created → {yaml_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  YOLOv8 Dataset Merger — ASEP Project")
    print("=" * 60)

    # Validate source datasets exist
    for tag, path in [("dataset1", DATASET1), ("dataset2", DATASET2)]:
        if not os.path.isdir(path):
            print(f"[✗] {tag} not found at: {path}")
            return
    print(f"[✓] dataset1 found → {DATASET1}")
    print(f"[✓] dataset2 found → {DATASET2}")

    # Clean & create output structure
    if os.path.exists(MERGED):
        print("[…] Removing existing dataset_merged/")
        shutil.rmtree(MERGED)

    ensure_dirs(MERGED, SPLITS)
    print(f"[✓] Created output tree → {MERGED}")
    print()

    # ── Copy & merge ──────────────────────────
    summary = {}

    for split in SPLITS:
        print(f"── {split.upper()} ──")

        img_dst = os.path.join(MERGED, split, "images")
        lbl_dst = os.path.join(MERGED, split, "labels")

        # Dataset 1
        ds1_img = copy_files(os.path.join(DATASET1, split, "images"), img_dst, "ds1_")
        ds1_lbl = copy_files(os.path.join(DATASET1, split, "labels"), lbl_dst, "ds1_")
        print(f"  dataset1 → {ds1_img} images, {ds1_lbl} labels")

        # Dataset 2
        ds2_img = copy_files(os.path.join(DATASET2, split, "images"), img_dst, "ds2_")
        ds2_lbl = copy_files(os.path.join(DATASET2, split, "labels"), lbl_dst, "ds2_")
        print(f"  dataset2 → {ds2_img} images, {ds2_lbl} labels")

        total_img = ds1_img + ds2_img
        total_lbl = ds1_lbl + ds2_lbl
        summary[split] = {"images": total_img, "labels": total_lbl}
        print(f"  merged   → {total_img} images, {total_lbl} labels")
        print()

    # ── Fix all labels to class 0 ─────────────
    print("[…] Fixing label class IDs → 0 (ambulance)")
    for split in SPLITS:
        fix_labels(os.path.join(MERGED, split, "labels"))
    print("[✓] All labels normalized to class 0")
    print()

    # ── Create data.yaml ──────────────────────
    create_data_yaml(MERGED)
    print()

    # ── Final summary ─────────────────────────
    print("=" * 60)
    print("  MERGE COMPLETE — Summary")
    print("=" * 60)
    grand_total = 0
    for split in SPLITS:
        s = summary[split]
        print(f"  {split:>6}: {s['images']:>6} images | {s['labels']:>6} labels")
        grand_total += s["images"]
    print(f"  {'TOTAL':>6}: {grand_total:>6} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
