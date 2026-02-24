# 🤟 Sign Language Translator — YOLOv8 + CNN

Real-time American Sign Language (ASL) recognition using a dual-model pipeline — YOLOv8 for hand detection and a custom CNN for gesture classification. Supports RGB, HSV, and Grayscale color spaces for adaptive performance across varying lighting conditions.

> 📦 Dataset available on Kaggle: [ASL Sign Language Dataset — RGB, HSV & Grayscale](https://www.kaggle.com/datasets/rudrakumargupta/asl-sign-language-dataset-rgb-hsv-and-grayscale)


> 🤖 Pretrained models available on Kaggle: *(add link after uploading models)*
> 📦 Models available on Kaggle: [ASL Sign Language CNN Models](https://www.kaggle.com/models/rudrakumargupta/asl-sign-language-cnn-models)
---

## 🚀 Demo

<img width="300" height="200" alt="Screenshot 2026-02-20 154352" src="https://github.com/user-attachments/assets/605556d0-1857-4712-aaf4-9f604f875f73" />
<img width="300" height="200" alt="Screenshot 2025-11-07 173211" src="https://github.com/user-attachments/assets/cb598bc4-1890-49b4-ad84-c186b4aeff60" />
<img width="300" height="200" alt="Screenshot 2025-11-07 172820" src="https://github.com/user-attachments/assets/f18ae3a6-49b7-477f-92eb-c3cdc10d5a5e" />

---

## 🧠 How It Works

1. **YOLOv8** detects and crops the hand region from live webcam feed
2. **CNN** classifies the gesture in the selected color space (RGB / HSV / Grayscale)
3. Recognized sign is displayed on screen in real time

---

## 🎨 Color Space Comparison

| Mode | FPS | Best For |
|---|---|---|
| RGB | ~20 FPS | Color-rich environments, highest accuracy |
| HSV | ~35 FPS | Varying lighting conditions |
| Grayscale | ~45 FPS | Speed, low-resource systems |

---

## 📁 Project Structure

```
Sign_language_Translator/
├── app.py                  # Main application — run this
├── run_app.bat             # Windows batch launcher
├── datasets/               # ASL dataset (RGB, HSV, Grayscale splits) on Kaggle
│   ├── rgb/
│   ├── hsv/
│   └── grey/
├── models/                 # Pretrained CNN weights on kaggle
│   ├── rgb_best.pt
│   ├── hsv_best.pt
│   └── grey_best.pt
└── scripts/                # Utility and training scripts
    ├── train_yolo.py       # YOLO training
    ├── split_train_val.py  # Dataset splitting
    ├── check_and_clean_labels.py
    ├── resume_train.py
    ├── overfit_test.py
    ├── gt_overlayer.py
    └── verify_setup.py

```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/rudrakumargupta/sign-language-translator.git
cd sign-language-translator
```

### 2. Install dependencies
```bash
pip install torch torchvision opencv-python ultralytics numpy pillow
```

### 3. Download pretrained models
Download from [Kaggle *(https://www.kaggle.com/models/rudrakumargupta/asl-sign-language-cnn-models)*] and place `.pt` files in the `models/` folder.

### 4. Run the app
```bash
python app.py
```
Or on Windows:
```bash
run_app.bat
```

---

## 🗃️ Dataset

The ASL dataset is available on Kaggle in three preprocessed versions (RGB, HSV, Grayscale) with train/val splits.

🔗 [ASL Sign Language Dataset — RGB, HSV & Grayscale](https://www.kaggle.com/datasets/rudrakumargupta/asl-sign-language-dataset-rgb-hsv-and-grayscale)

---

## 📋 Requirements

- Python 3.8+
- Webcam or video input source
- Pretrained model weights (see Setup)

---

## 🛠️ Tech Stack

`Python` · `PyTorch` · `YOLOv8` · `OpenCV` · `CNN` · `Ultralytics`

---

## 👤 Author

**Rudra Kumar Gupta**
Electronics Engineering Student · AI/ML Intern · Game Enthusiast

🔗 [Kaggle](https://www.kaggle.com/rudrakumargupta) · [GitHub](https://github.com/rudrakumargupta)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
