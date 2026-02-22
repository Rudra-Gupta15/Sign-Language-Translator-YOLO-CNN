# preview_labels.py
from pathlib import Path
import cv2

ROOT = Path(r"C:/Users/shriv/stl_rgb/final_dataset")  # <- change if needed
OUT  = ROOT/"_preview_gt"
OUT.mkdir(exist_ok=True)

def draw_box(img, xc,yc,w,h, color=(0,255,0)):
    H,W = img.shape[:2]
    x1 = int((xc - w/2)*W); y1 = int((yc - h/2)*H)
    x2 = int((xc + w/2)*W); y2 = int((yc + h/2)*H)
    cv2.rectangle(img,(x1,y1),(x2,y2),color,2)

def preview(split="train", limit=50):
    imdir = ROOT/f"{split}/images"
    lbdir = ROOT/f"{split}/labels"
    shown = 0
    for imgp in imdir.glob("*.*"):
        lblp = lbdir/(imgp.stem + ".txt")
        if not lblp.exists(): continue
        img = cv2.imread(str(imgp))
        ok=True
        for ln in lblp.read_text().strip().splitlines():
            p = ln.split()
            if len(p) != 5: ok=False; break
            cls, xc, yc, w, h = int(float(p[0])), *map(float,p[1:])
            if not (0<=xc<=1 and 0<=yc<=1 and 0<w<=1 and 0<h<=1): ok=False; break
            draw_box(img, xc,yc,w,h)
        color = (0,255,0) if ok else (0,0,255)
        if not ok: cv2.putText(img,"BAD LABEL", (10,30), cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
        cv2.imwrite(str(OUT/f"{split}_{imgp.name}"), img)
        shown+=1
        if shown>=limit: break

preview("train", 50)
preview("val", 30)
print("Wrote previews to", OUT)
