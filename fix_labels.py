"""
ASEP — Dataset Label Fixer
============================
Fixes YOLO label class IDs. Creates backups before modifying.
"""
import os, glob, shutil

def fix_labels(folder, src=1, dst=0, backup=True):
    if not os.path.exists(folder):
        print(f"[SKIP] Not found: {folder}")
        return 0
    files = glob.glob(f"{folder}/**/*.txt", recursive=True)
    fixed = 0
    for f in files:
        lines = open(f).readlines()
        new, changed = [], False
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0] == str(src):
                parts[0] = str(dst)
                changed = True
            new.append(' '.join(parts))
        if changed:
            if backup:
                shutil.copy2(f, f + '.bak')
            open(f, 'w').write('\n'.join(new))
            fixed += 1
    print(f"[FIX] {folder}: {fixed}/{len(files)} fixed")
    return fixed

if __name__ == '__main__':
    print("ASEP — Label Fixer (class 1 → 0)")
    base = os.path.join(os.path.dirname(__file__), "dataset")
    t = 0
    for split in ["train", "valid", "test"]:
        t += fix_labels(os.path.join(base, split, "labels"))
    print(f"Total fixed: {t}")
