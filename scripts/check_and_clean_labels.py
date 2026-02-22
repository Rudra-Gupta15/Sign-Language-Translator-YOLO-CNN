from pathlib import Path
import math

# CHANGE this to your dataset root that contains train/ and val/
ROOT = Path(r"C:/Users/shriv/stl_rgb/final_dataset")
NC = 36  # number of classes

def is_float(x):
    try:
        float(x)
        return True
    except:
        return False

bad_files = []
fixed_files = []
seg_files = []

def check_dir(lbl_dir):
    global bad_files, fixed_files, seg_files
    for p in lbl_dir.glob("*.txt"):
        lines = p.read_text().strip().splitlines()
        new_lines = []
        changed = False
        is_seg_file = False

        for ln in lines:
            parts = ln.strip().split()
            if len(parts) == 0:
                continue

            # If it's segmentation (len > 5), mark and SKIP this line for detect training
            if len(parts) > 5:
                is_seg_file = True
                # try to convert polygon -> bbox if you want. For now, drop it:
                changed = True
                continue

            if len(parts) != 5:
                bad_files.append((p, f"wrong column count: {len(parts)}"))
                continue

            cls, xc, yc, w, h = parts
            # numeric checks
            if not (cls.isdigit() and all(is_float(x) for x in [xc, yc, w, h])):
                bad_files.append((p, f"non-numeric: {parts}"))
                continue

            cls = int(cls)
            xc, yc, w, h = map(float, [xc, yc, w, h])

            # NaN/inf checks
            if any([math.isnan(x) or math.isinf(x) for x in [xc, yc, w, h]]):
                bad_files.append((p, "nan/inf values"))
                continue

            # class in range
            if not (0 <= cls < NC):
                bad_files.append((p, f"class out of range: {cls}"))
                continue

            # coords in [0,1], positive size (allow tiny but not <=0)
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_files.append((p, f"coords out of [0,1] or zero/neg size: {xc,yc,w,h}"))
                continue

            new_lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        if is_seg_file:
            seg_files.append(p)

        if changed or len(new_lines) != len(lines):
            # overwrite file with only valid detect lines
            p.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
            fixed_files.append(p)

# Run checks on both train and val
check_dir(ROOT / "train" / "labels")
check_dir(ROOT / "val" / "labels")

print(f"Fixed/rewritten label files: {len(fixed_files)}")
print(f"Files that had segmentation lines removed: {len(seg_files)}")
print(f"Files still flagged bad: {len(bad_files)}")
for f, reason in bad_files[:20]:
    print("  -", f, "->", reason)
if len(bad_files) > 20:
    print("  ... (showing first 20)")
