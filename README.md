# Sign Language Translator - YOLO & CNN

Real-time sign language recognition using YOLOv8 hand detection + CNN gesture classification. Supports RGB, Grayscale, and HSV color spaces for adaptive performance across varying lighting conditions.

## Setup

Install dependencies:
```bash
pip install torch torchvision opencv-python ultralytics numpy pillow
```

Run:
```bash
python app.py
```

## How It Works

1. YOLO detects hands in real-time video
2. CNN classifies gestures in selected color space (RGB/Grayscale/HSV)
3. Displays recognized signs on screen

## Features

- **Dual-Model Architecture**: YOLO + CNN
- **Multi-Color Space**: RGB (color-rich), Grayscale (fast), HSV (lighting-robust)
- **Real-Time**: ~20-45 FPS depending on color space
- **Adaptive**: Handles varying lighting and backgrounds

## Requirements

- Python 3.8+
- Pre-trained model weights (download separately and configure paths in `app.py`)
- Webcam/video input source

## License

MIT License
