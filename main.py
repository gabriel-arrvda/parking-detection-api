from ultralytics import YOLO

model = YOLO('model/best.pt')
model.predict(source="sources/parking.webp", imgsz=640, conf=0.6, save=True)