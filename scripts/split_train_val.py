import os, shutil, random, glob
from pathlib import Path

ROOT = Path(r"C:\Users\shriv\stl_rgb\final_dataset") 
IMGS = list(glob.glob(str(ROOT/"train/images/*.*")))
random.shuffle(IMGS)

val_ratio = 0.1
val_n = int(len(IMGS)*val_ratio)

VAL_IMGS = set(IMGS[:val_n])

(ROOT/"val/images").mkdir(parents=True, exist_ok=True)
(ROOT/"val/labels").mkdir(parents=True, exist_ok=True)

for img_path in VAL_IMGS:
    img = Path(img_path)
    lbl = ROOT/"train/labels"/(img.stem + ".txt")
    shutil.move(img_path, ROOT/"val/images"/img.name)
    if lbl.exists():
        shutil.move(lbl, ROOT/"val/labels"/lbl.name)
