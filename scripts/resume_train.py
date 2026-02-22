from ultralytics import YOLO
m = YOLO(r"runs_local/36cls_yolov8n9/weights/last.pt")  # change if your run name differs
m.train(data=r"C:/Users/shriv/stl_rgb/final_dataset/data.yaml", resume=True, workers=0)
