# train_yolov8s.py
from ultralytics import YOLO
import os

# ====== USER SETTINGS (edit these) ======
DATA_YAML = r"C:/Users/shriv/stl_rgb/final_dataset/data.yaml"  # your data.yaml
MODEL     = "yolov8s.pt"        # start with small model
IMG_SIZE  = 512                 # 512 if VRAM is tight
EPOCHS    = 150                  # 120–200 is common
BATCH     = 4                  # try 8 if 640->512
WORKERS   = 0                    # Windows-friendly
RUNS_DIR  = "runs_local"         # output root
RUN_NAME  = "36cls_yolov8s_640"  # change per experiment
SEED      = 42
# =======================================

# Disable online loggers like wandb
os.environ["WANDB_DISABLED"] = "true"

def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        project=RUNS_DIR,
        name=RUN_NAME,
        seed=SEED,
        # stability & efficiency
        amp=False,          # set False only if you hit NaNs
        rect=True,         # better packing for varied aspect ratios
        cos_lr=True,
        patience=40,       # early stop patience
        # moderate augmentations (tune later)
        mosaic=0.5,
        mixup=0.0,
        auto_augment="randaugment",
        erasing=0.2,
        # misc
        verbose=True
    )
    print(f"\n✅ Training complete. See: {RUNS_DIR}/{RUN_NAME}")

if __name__ == "__main__":
    main()
