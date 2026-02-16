import os
import sys
import threading
import time
from collections import deque
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Try PyTorch to decide device for YOLO
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# Optional backends
YOLO_AVAILABLE = True
KERAS_AVAILABLE = True

try:
    from ultralytics import YOLO
except Exception:
    YOLO_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    KERAS_AVAILABLE = False


# =========================
# Configuration
# =========================
MODEL_PATHS = {
    "RGB": r"models\rgb_best.pt",
    "GREYSCALE": r"models\grey_best.pt",
    "HSV": r"models\hsv_best.pt"
}

DEFAULT_CAM_INDEX = 0
DEFAULT_CONF = 0.25
SMOOTHING_WINDOW = 8  # Slightly reduced for responsiveness
IMGSZ_YOLO = 640

# Keras defaults
KERAS_INPUT_SIZE = (224, 224)
KERAS_COLOR_MODE = "rgb"
KERAS_RESCALE = 1.0 / 255.0

# UI Colors (Beige & White Theme)
COLOR_BG = "#F5F5DC"        # Beige
COLOR_PANEL = "#FFFFFF"     # White
COLOR_ACCENT = "#8D6E63"    # Warm Brown Accent
COLOR_TEXT = "#3E2723"      # Dark Brown Text (High Contrast)
COLOR_TEXT_DIM = "#795548"  # Lighter Brown
COLOR_SUCCESS = "#8D6E63"   # Brown Accent
COLOR_ERROR = "#D32F2F"     # Standard Red (for visibility)
COLOR_WARNING = "#FBC02D"   # Mustard Yellow

FONT_HEADER = ("Segoe UI", 24, "bold")
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_NORMAL = ("Segoe UI", 11)
FONT_SMALL = ("Segoe UI", 9)


def load_labels_txt(model_path):
    """Look for labels.txt next to the .h5 and return a list or None."""
    base_dir = os.path.dirname(model_path)
    labels_path = os.path.join(base_dir, "labels.txt")
    if os.path.isfile(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        return names
    return None


class BackendType:
    YOLO = "yolo"
    KERAS = "keras"


class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        
        # === AUTO-FOCUS ATTEMPT ===
        try:
            self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 1) # Turn ON
            self.stream.set(cv2.CAP_PROP_FOCUS, 0)     # Trigger
        except Exception:
            pass
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.get, args=(), daemon=True).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (grabbed, frame) = self.stream.read()
                with self.lock:
                    self.grabbed = grabbed
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.grabbed, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class HomePage:
    """Modern dark-themed landing page."""
    
    def __init__(self, master, on_start_callback):
        self.master = master
        self.on_start_callback = on_start_callback
        self.frame = tk.Frame(master, bg=COLOR_BG)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        # Center container
        container = tk.Frame(self.frame, bg=COLOR_PANEL,  padx=60, pady=60)
        container.place(relx=0.5, rely=0.5, anchor="center")
        
        # Logo/Icon
        tk.Label(container, text="🤟", bg=COLOR_PANEL, fg=COLOR_TEXT, 
                 font=("Segoe UI Emoji", 64)).pack(pady=(0, 20))
        
        # Title
        tk.Label(container, text="SIGN LANGUAGE TRANSLATOR", bg=COLOR_PANEL, fg=COLOR_TEXT,
                 font=("Segoe UI", 32, "bold")).pack()
        
        tk.Label(container, text="AI-Powered Real-Time Recognition", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=("Segoe UI", 14)).pack(pady=(0, 40))
        
        # Features Grid
        features_frame = tk.Frame(container, bg=COLOR_PANEL)
        features_frame.pack(pady=20)
        
        features = ["🚀 High Performance", "🎯 Auto-Focus & Tracking", "🎥 Smooth Video"]
        for feat in features:
            lbl = tk.Label(features_frame, text=feat, bg=COLOR_PANEL, fg=COLOR_TEXT_DIM,
                          font=("Segoe UI", 12))
            lbl.pack(side=tk.LEFT, padx=20)
            
        # Divider
        tk.Frame(container, bg=COLOR_ACCENT, height=2, width=400).pack(pady=30)
        
        # Start Button
        self.start_btn = tk.Button(container, text="LAUNCH APPLICATION", 
                                  bg=COLOR_ACCENT, fg="#000000",
                                  font=("Segoe UI", 12, "bold"),
                                  relief=tk.FLAT, padx=40, pady=15,
                                  cursor="hand2", command=self._on_start)
        self.start_btn.pack()
        
        # Hover effects
        self.start_btn.bind("<Enter>", lambda e: self.start_btn.config(bg="#A1887F"))
        self.start_btn.bind("<Leave>", lambda e: self.start_btn.config(bg=COLOR_ACCENT))

    def _on_start(self):
        self.frame.destroy()
        self.on_start_callback()


class SignTranslatorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Sign Language Translator Pro")
        self.master.geometry("1400x900")
        self.master.configure(bg=COLOR_BG)
        self.master.protocol("WM_DELETE_WINDOW", self.on_close)

        # Application State
        self.video_getter = None
        self.running = False
        self.model = None
        self.backend = None
        self.labels = None
        self.current_model_type = None
        
        self.last_inference_result = None
        self.last_inference_conf = 0.0
        self.last_inference_box = None # For tracking
        self.inference_lock = threading.Lock()
        
        # Settings
        self.conf_threshold = 0.45  # INCREASED: Stricter checking (was 0.25)
        self.use_roi = tk.BooleanVar(value=True)  # ROI Toggle
        self.cam_index = DEFAULT_CAM_INDEX
        
        # Tracking State
        self.roi_center_x = 0.5
        self.roi_center_y = 0.5
        
        # Sentence Building
        self.sentence = ""
        self.buffer = deque(maxlen=SMOOTHING_WINDOW)
        self.last_stable_pred = None
        self.stable_start_time = None
        self.HOLD_TIME = 3.0  # INCREASED: Longer hold time (was 2.0)
        
        self._setup_styles()
        self._build_ui()
        
        # Start the GUI update loop (Animation & Video)
        self._gui_loop()
        
        # Start Inference Thread (runs in background)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Colors
        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT, font=FONT_NORMAL)
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL)
        
        # Buttons
        style.configure("Accent.TButton", background=COLOR_ACCENT, foreground="black", 
                       font=("Segoe UI", 10, "bold"), borderwidth=0, padding=10)
        style.map("Accent.TButton", background=[("active", "#A370DB")])
        
        style.configure("Stop.TButton", background=COLOR_ERROR, foreground="black",
                       font=("Segoe UI", 10, "bold"), borderwidth=0, padding=10)
        style.map("Stop.TButton", background=[("active", "#B54D60")])
        
        # Toggle
        style.configure("Switch.TCheckbutton", background=COLOR_PANEL, foreground=COLOR_TEXT,
                       font=FONT_NORMAL)

    def _build_ui(self):
        # Main Layout: Sidebar (Left) + Content (Right)
        
        # === Sidebar ===
        sidebar = ttk.Frame(self.master, style="Panel.TFrame", width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # App Header in Sidebar
        header_frame = tk.Frame(sidebar, bg=COLOR_PANEL, pady=20)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="SETTINGS", bg=COLOR_PANEL, fg=COLOR_ACCENT,
                 font=FONT_TITLE).pack()

        # 1. Model Selection
        self._build_sidebar_section(sidebar, "🧠 AI MODELS")
        
        btn_frame = tk.Frame(sidebar, bg=COLOR_PANEL)
        btn_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.btn_rgb = self._make_sidebar_btn(btn_frame, "🔴 RGB Model", lambda: self.load_preset("RGB"))
        self.btn_grey = self._make_sidebar_btn(btn_frame, "⚫ Grey Model", lambda: self.load_preset("GREYSCALE"))
        self.btn_hsv = self._make_sidebar_btn(btn_frame, "🌈 HSV Model", lambda: self.load_preset("HSV"))
        
        ttk.Separator(sidebar, orient='horizontal').pack(fill=tk.X, padx=20, pady=15)

        # 2. Camera Controls
        self._build_sidebar_section(sidebar, "📷 CAMERA")
        
        cam_frame = tk.Frame(sidebar, bg=COLOR_PANEL)
        cam_frame.pack(fill=tk.X, padx=15, pady=5)
        
        self.btn_start = ttk.Button(cam_frame, text="▶ START CAMERA", style="Accent.TButton",
                                   command=self.start_camera)
        self.btn_start.pack(fill=tk.X, pady=5)
        
        self.btn_stop = ttk.Button(cam_frame, text="⏹ STOP CAMERA", style="Stop.TButton",
                                  command=self.stop_camera)
        self.btn_stop.pack(fill=tk.X, pady=5)
        self.btn_stop.state(['disabled'])
        
        # 3. Accuracy/Focus
        self._build_sidebar_section(sidebar, "🎯 ACCURACY")
        
        opts_frame = tk.Frame(sidebar, bg=COLOR_PANEL)
        opts_frame.pack(fill=tk.X, padx=15, pady=5)
        
        # ROI Toggle
        ttk.Checkbutton(opts_frame, text="Show Focus Box", variable=self.use_roi, 
                       style="Switch.TCheckbutton").pack(anchor="w", pady=5)
        
        tk.Label(opts_frame, text="Confidence Threshold:", bg=COLOR_PANEL, fg=COLOR_TEXT_DIM,
                 font=FONT_SMALL).pack(anchor="w", pady=(10, 0))
        
        self.conf_scale = tk.Scale(opts_frame, from_=0.1, to=1.0, resolution=0.05, 
                                  orient=tk.HORIZONTAL, bg=COLOR_PANEL, fg=COLOR_ACCENT,
                                  highlightthickness=0, troughcolor="#333333",
                                  command=self._on_conf_change)
        self.conf_scale.set(DEFAULT_CONF)
        self.conf_scale.pack(fill=tk.X)
        self.lbl_conf = tk.Label(opts_frame, text=f"{DEFAULT_CONF:.2f}", bg=COLOR_PANEL, fg=COLOR_ACCENT)
        self.lbl_conf.pack(anchor="e")

        # === Main Content Area ===
        content = tk.Frame(self.master, bg=COLOR_BG)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Top Bar: Status & Sentence
        top_bar = tk.Frame(content, bg=COLOR_BG)
        top_bar.pack(fill=tk.X, pady=(0, 20))
        
        # Status
        self.lbl_status = tk.Label(top_bar, text="Ready", bg=COLOR_BG, fg=COLOR_TEXT_DIM,
                                  font=FONT_SMALL)
        self.lbl_status.pack(side=tk.LEFT)
        
        # Video Feed
        self.video_frame = tk.Frame(content, bg="#000000")
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.video_frame.bind("<Configure>", self._on_resize)
        
        self.lbl_video = tk.Label(self.video_frame, bg="#000000", text="Camera Offline",
                                 fg="#555555", font=("Segoe UI", 20))
        self.lbl_video.pack(fill=tk.BOTH, expand=True)
        
        # Bottom Bar: Prediction & Sentence Display
        bottom_panel = tk.Frame(content, bg=COLOR_PANEL, height=150, pady=15, padx=20)
        bottom_panel.pack(fill=tk.X, pady=(20, 0))
        bottom_panel.pack_propagate(False)
        
        # Current Prediction (Large)
        pred_box = tk.Frame(bottom_panel, bg=COLOR_PANEL)
        pred_box.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 30))
        
        tk.Label(pred_box, text="DETECTED", bg=COLOR_PANEL, fg=COLOR_TEXT_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        self.lbl_pred = tk.Label(pred_box, text="--", bg=COLOR_PANEL, fg=COLOR_SUCCESS,
                                font=("Segoe UI", 36, "bold"))
        self.lbl_pred.pack(anchor="w")
        
        # Sentence Construction
        sent_box = tk.Frame(bottom_panel, bg=COLOR_PANEL)
        sent_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(sent_box, text="SENTENCE", bg=COLOR_PANEL, fg=COLOR_TEXT_DIM,
                 font=FONT_SMALL).pack(anchor="w")
        
        self.entry_sentence = tk.Entry(sent_box, bg="#2D2D2D", fg="white", 
                                      font=("Segoe UI", 18), relief=tk.FLAT,
                                      insertbackground="white")
        self.entry_sentence.pack(fill=tk.X, pady=5, ipady=5)
        
        # Sentence Controls
        ctrl_box = tk.Frame(sent_box, bg=COLOR_PANEL)
        ctrl_box.pack(fill=tk.X)
        
        ttk.Button(ctrl_box, text="SPACE", command=lambda: self._edit_sentence("space")).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ctrl_box, text="BACKSPACE", command=lambda: self._edit_sentence("back")).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_box, text="CLEAR", command=lambda: self._edit_sentence("clear")).pack(side=tk.LEFT, padx=5)

    def _build_sidebar_section(self, parent, title):
        frame = tk.Frame(parent, bg=COLOR_PANEL, pady=5)
        frame.pack(fill=tk.X, padx=15, pady=(15, 5))
        tk.Label(frame, text=title, bg=COLOR_PANEL, fg=COLOR_TEXT, 
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")

    def _make_sidebar_btn(self, parent, text, command):
        btn = tk.Button(parent, text=text, bg=COLOR_PANEL, fg=COLOR_TEXT,
                       font=("Segoe UI", 10), relief=tk.FLAT, anchor="w",
                       padx=10, pady=8, command=command, cursor="hand2")
        btn.pack(fill=tk.X, pady=2)
        
        def on_enter(e):
            if btn['bg'] != COLOR_ACCENT:
                btn['bg'] = "#2D2D2D"
        def on_leave(e):
            if btn['bg'] != COLOR_ACCENT:
                btn['bg'] = COLOR_PANEL
                
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def _on_resize(self, event):
        # Handle video resizing if needed
        pass

    def _on_conf_change(self, val):
        self.conf_threshold = float(val)
        self.lbl_conf.config(text=f"{self.conf_threshold:.2f}")

    def load_preset(self, model_type):
        """Load a model preset."""
        path = MODEL_PATHS.get(model_type)
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", f"Model file not found:\n{path}")
            return
            
        self._load_model_file(path, model_type)
        
        # Update UI buttons
        for btn in [self.btn_rgb, self.btn_grey, self.btn_hsv]:
            btn.configure(bg=COLOR_PANEL, fg=COLOR_TEXT)
            
        if model_type == "RGB": self.btn_rgb.configure(bg=COLOR_ACCENT, fg="black")
        if model_type == "GREYSCALE": self.btn_grey.configure(bg=COLOR_ACCENT, fg="black")
        if model_type == "HSV": self.btn_hsv.configure(bg=COLOR_ACCENT, fg="black")

    def _load_model_file(self, path, type_name):
        self.lbl_status.config(text=f"Loading {type_name} model...", fg=COLOR_WARNING)
        self.master.update()
        
        try:
            if path.endswith(".pt"):
                # YOLO
                if not YOLO_AVAILABLE:
                    raise ImportError("Ultralytics not installed")
                model = YOLO(path)
                self.backend = BackendType.YOLO
                # Extract names
                if hasattr(model, "names"):
                    self.labels = model.names
                else:
                    self.labels = None
                self.model = model
                
            elif path.endswith(".h5"):
                # Keras
                if not KERAS_AVAILABLE:
                    raise ImportError("TensorFlow not installed")
                self.model = keras_load_model(path)
                self.labels = load_labels_txt(path)
                self.backend = BackendType.KERAS
                
            self.current_model_type = type_name
            self.lbl_status.config(text=f"Loaded {type_name} ({self.backend})", fg=COLOR_SUCCESS)
            
        except Exception as e:
            self.lbl_status.config(text="Model Load Failed", fg=COLOR_ERROR)
            messagebox.showerror("Load Error", str(e))

    def start_camera(self):
        if not self.model:
            messagebox.showwarning("No Model", "Please select a model first!")
            return
            
        if self.video_getter:
            self.video_getter.stop()
            
        try:
            self.video_getter = VideoGet(self.cam_index).start()
            self.running = True
            
            self.btn_start.state(['disabled'])
            self.btn_stop.state(['!disabled'])
            self.lbl_status.config(text="Camera Active - Inference Running", fg=COLOR_SUCCESS)
            
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))

    def stop_camera(self):
        self.running = False
        if self.video_getter:
            self.video_getter.stop()
            self.video_getter = None
            
        self.btn_start.state(['!disabled'])
        self.btn_stop.state(['disabled'])
        self.lbl_status.config(text="Camera Stopped", fg=COLOR_TEXT_DIM)
        self.lbl_video.configure(image='', text="Camera Offline")

    def _inference_loop(self):
        """Background thread for heavy AI processing."""
        while True:
            if not self.running or not self.video_getter:
                time.sleep(0.1)
                continue
                
            grabbed, frame = self.video_getter.read()
            if not grabbed or frame is None:
                continue

            # === ROI TRACKING LOGIC ===
            h, w = frame.shape[:2]
            box_size = int(min(h, w) * 0.6)
            
            # Smoothly update ROI center
            cx = int(self.roi_center_x * w)
            cy = int(self.roi_center_y * h)
            
            # Clamp
            half_box = box_size // 2
            cx = max(half_box, min(w - half_box, cx))
            cy = max(half_box, min(h - half_box, cy))
            
            x1 = cx - half_box
            y1 = cy - half_box
            x2 = cx + half_box
            y2 = cy + half_box
            
            # Crop if ROI is enabled for inference
            if self.use_roi.get():
                inf_frame = frame[y1:y2, x1:x2]
            else:
                inf_frame = frame

            try:
                # === INFERENCE ===
                detected_cls = None
                conf = 0.0
                detected_box = None # Relative to inf_frame
                
                if self.backend == BackendType.YOLO:
                    results = self.model.predict(
                        source=inf_frame,
                        conf=self.conf_threshold,
                        imgsz=IMGSZ_YOLO,
                        device=DEVICE,
                        verbose=False
                    )
                    if len(results) > 0 and results[0].boxes:
                        box = results[0].boxes[0] # Top result
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        
                        # Store box for tracking (x1, y1, x2, y2)
                        detected_box = box.xyxy[0].cpu().numpy()
                        
                        if self.labels and isinstance(self.labels, dict):
                            detected_cls = self.labels.get(cls_id, str(cls_id))
                        elif self.labels and isinstance(self.labels, list):
                            detected_cls = self.labels[cls_id] if cls_id < len(self.labels) else str(cls_id)
                        else:
                            detected_cls = str(cls_id)

                elif self.backend == BackendType.KERAS:
                    # Preprocess
                    img = cv2.cvtColor(inf_frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, KERAS_INPUT_SIZE)
                    if KERAS_COLOR_MODE == "grayscale":
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                        img = img[..., np.newaxis]
                    
                    img = img.astype("float32") * KERAS_RESCALE
                    batch = np.expand_dims(img, axis=0)
                    
                    probs = self.model.predict(batch, verbose=0)[0]
                    cls_id = np.argmax(probs)
                    conf = float(probs[cls_id])
                    
                    if conf > self.conf_threshold:
                        if self.labels:
                            detected_cls = self.labels[cls_id]
                        else:
                            detected_cls = str(cls_id)
                
                # Update shared state
                with self.inference_lock:
                    self.last_inference_result = detected_cls
                    self.last_inference_conf = conf
                    
                    # Update ROI Target if detection found
                    if detected_box is not None and self.use_roi.get():
                        # Box center relative to ROI
                        bx1, by1, bx2, by2 = detected_box
                        bcx = (bx1 + bx2) / 2
                        bcy = (by1 + by2) / 2
                        
                        # Convert to absolute frame coordinates
                        abs_cx = x1 + bcx
                        abs_cy = y1 + bcy
                        
                        # Update target (simple low-pass filter for smoothness)
                        alpha = 0.1 # Smoothing factor
                        self.roi_center_x = self.roi_center_x * (1 - alpha) + (abs_cx / w) * alpha
                        self.roi_center_y = self.roi_center_y * (1 - alpha) + (abs_cy / h) * alpha

            except Exception as e:
                print(f"Inference Error: {e}")

                
            # Limit inference rate slightly to save CPU if needed
            # time.sleep(0.01)

    def _gui_loop(self):
        """Main thread loop for GUI updates and Video rendering."""
        if self.running and self.video_getter:
            grabbed, frame = self.video_getter.read()
            if grabbed and frame is not None:
                
                # Get latest inference result
                with self.inference_lock:
                    res = self.last_inference_result
                    conf = self.last_inference_conf
                
                # 1. Process Smoothing
                if res and conf > self.conf_threshold:
                    self.buffer.append(res)
                else:
                    self.buffer.append(None)
                
                # Most common in buffer
                clean_pred = None
                if len(self.buffer) == self.buffer.maxlen:
                    valid_preds = [x for x in self.buffer if x]
                    if valid_preds:
                        # Find mode
                        clean_pred = max(set(valid_preds), key=valid_preds.count)
                
                # 2. Update Prediction Labels
                if clean_pred:
                    self.lbl_pred.config(text=clean_pred, fg=COLOR_SUCCESS)
                    self._check_sentence_building(clean_pred)
                else:
                    self.lbl_pred.config(text="--", fg=COLOR_TEXT_DIM)
                    
                
                # 3. Draw Overlays (ROI, Text)
                frame_draw = frame.copy()
                h, w = frame_draw.shape[:2]
                
                # Draw ROI Box
                if self.use_roi.get():
                    box_size = int(min(h, w) * 0.6)
                    
                    # Calculate current ROI position based on tracked center
                    cx = int(self.roi_center_x * w)
                    cy = int(self.roi_center_y * h)
                    half_box = box_size // 2
                    
                    x1 = cx - half_box
                    y1 = cy - half_box
                    x2 = cx + half_box
                    y2 = cy + half_box
                    
                    # Color based on detection
                    color = (0, 255, 0) if clean_pred else (200, 200, 200)
                    thickness = 4 if clean_pred else 2
                    
                    cv2.rectangle(frame_draw, (x1, y1), (x2, y2), color, thickness)
                    
                    label_text = "TRACKING..." if clean_pred else "PLACE HAND HERE"
                    cv2.putText(frame_draw, label_text, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw Live Confidence
                if res:
                    cv2.putText(frame_draw, f"{res} ({conf:.2f})", (30, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # 4. Display in Tkinter
                # Resize to fit video frame
                canvas_w = self.video_frame.winfo_width()
                canvas_h = self.video_frame.winfo_height()
                
                if canvas_w > 10 and canvas_h > 10: # Ensure valid size
                    scale = min(canvas_w/w, canvas_h/h)
                    nw, nh = int(w*scale), int(h*scale)
                    resized = cv2.resize(frame_draw, (nw, nh))
                    
                    # Convert to RGB for PIL
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.lbl_video.imgtk = imgtk # Keep reference
                    self.lbl_video.configure(image=imgtk, text="")
        
        # Schedule next update (approx 30 FPS)
        self.master.after(30, self._gui_loop)

    def _check_sentence_building(self, pred):
        """Hold-to-type logic."""
        if pred == self.last_stable_pred:
            if self.stable_start_time:
                elapsed = time.time() - self.stable_start_time
                if elapsed > self.HOLD_TIME:
                    # Append char
                    self._edit_sentence(pred)
                    self.stable_start_time = None # Reset
                    self.last_stable_pred = None # Require re-entry
                    self.buffer.clear() # Clear buffer to prevent double typing
                    # self._show_flash("Added!")
        else:
            self.last_stable_pred = pred
            self.stable_start_time = time.time()

    def _edit_sentence(self, action):
        current = self.sentence
        
        if action == "space":
            self.sentence += " "
        elif action == "back":
            self.sentence = self.sentence[:-1]
        elif action == "clear":
            self.sentence = ""
        else:
            # Append character
            self.sentence += action
            
        self.entry_sentence.delete(0, tk.END)
        self.entry_sentence.insert(0, self.sentence)

    def on_close(self):
        self.running = False
        if self.video_getter:
            self.video_getter.stop()
        self.master.destroy()


def main():
    root = tk.Tk()
    # Simple icon setup if available
    # root.iconbitmap("icon.ico")
    
    def launch():
        app = SignTranslatorApp(root)
        
    HomePage(root, launch)
    root.mainloop()

if __name__ == "__main__":
    main()