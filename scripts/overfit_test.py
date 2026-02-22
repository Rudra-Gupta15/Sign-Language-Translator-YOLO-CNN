from ultralytics import YOLO
import os

# === USER SETTINGS ===
DATA_YAML = r"C:/Users/shriv/stl_rgb/final_dataset/data.yaml"  # Path to your dataset config
MODEL     = "yolov8s.pt"        # Small model for quick overfit test
IMG_SIZE  = 512
EPOCHS    = 50
BATCH     = 4                   # Safe for GTX 1650
WORKERS   = 0                   # Windows safe
RUNS_DIR  = "runs_local"
RUN_NAME  = "overfit_test"
SEED      = 42

# Disable WANDB logging
os.environ["WANDB_DISABLED"] = "true"

def main():
    model = YOLO(MODEL)
    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        epochs=EPOCHS,
        fraction=0.01,           # Use only 1% of the dataset
        amp=False,               # Disable mixed precision to avoid NaNs
        rect=True,               # Keep aspect ratio
        mosaic=0.0,               # Disable mosaic augmentation
        mixup=0.0,                 # Disable mixup augmentation
        auto_augment="none",      # No automated augmentations
        erasing=0.0,              # Disable random erasing
        cache=False,              # Load fresh each time
        project=RUNS_DIR,
        name=RUN_NAME,
        seed=SEED,
        verbose=True
    )
    print(f"\n✅ Overfit test complete. Check results in: {RUNS_DIR}/{RUN_NAME}")

if __name__ == "__main__":
    main()
