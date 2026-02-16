# Sign Language Translator - YOLO & CNN

Real-time sign language recognition system using YOLO object detection and CNN for gesture classification. Features adaptive multi-color space processing (RGB, Grayscale, HSV) for robust performance across diverse lighting and environmental conditions.

## Features

- **Dual-Model Architecture**: YOLO for real-time hand detection + CNN for accurate gesture classification
- **Multi-Color Space Support**: Process video input in RGB, Grayscale, and HSV color spaces for optimal performance in any environment
- **Real-Time Processing**: Fast inference suitable for live video streams
- **Pre-trained Models**: Includes optimized models trained on ASL and sign language datasets
- **Environment-Adaptive**: Automatically handles varying lighting conditions and backgrounds

## Tech Stack

- **Detection**: YOLOv8 (Nano, Small variants)
- **Classification**: Convolutional Neural Networks (CNN)
- **Framework**: PyTorch
- **Computer Vision**: OpenCV
- **Language**: Python 3.8+

## Project Structure
```
Sign_lang/
├── app.py                      # Main application entry point
├── models/                     # Model weights
│   ├── rgb_best.pt            # RGB color space model
│   ├── grey_best.pt           # Grayscale color space model
│   ├── hsv_best.pt            # HSV color space model
│   └── pretrained/            # YOLO pre-trained weights
├── scripts/                    # Data processing utilities
├── logs/                       # Training logs and metrics
└── datasets/                   # Training datasets (RGB, Grey, HSV)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Sign-Language-Translator-YOLO-CNN.git
cd Sign_lang
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pre-trained models:
- Place model weights in `models/` directory

## Usage

Run the main application:
```bash
python app.py
```

Select your preferred color space (RGB, Grayscale, or HSV) for optimal performance in your environment.

## Model Performance

| Color Space | Accuracy | FPS |
|-------------|----------|-----|
| RGB        | [XX%]    | [XX] |
| Grayscale  | [XX%]    | [XX] |
| HSV        | [XX%]    | [XX] |

## How It Works

1. **Hand Detection**: YOLO detects hand regions in real-time video feed
2. **Color Space Processing**: Converts input to selected color space (RGB/Grayscale/HSV)
3. **Gesture Classification**: CNN classifies detected hand regions into sign language gestures
4. **Output**: Displays recognized sign or translates to text/speech

## Training

Models were trained on:
- ASL (American Sign Language) dataset - RGB
- Sign Language datasets - Grayscale and HSV variants

Training logs available in `logs/` directory.

## Advantages of Multi-Color Space Support

- **RGB**: Best for well-lit, color-rich environments
- **Grayscale**: Efficient for low-light conditions, reduces computational load
- **HSV**: Robust against lighting variations, ideal for inconsistent environments

## Future Improvements

- [ ] Add speech synthesis for text output
- [ ] Support for additional sign languages
- [ ] Mobile app deployment
- [ ] Batch processing for video files
- [ ] Gesture sequence recognition (multi-word sentences)

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- YOLO community for object detection models
- ASL dataset contributors
- OpenCV and PyTorch communities

---

**Note**: Replace `[XX%]` and `[XX]` with your actual model performance metrics.
