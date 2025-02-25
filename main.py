from ultralytics import YOLO

# model = YOLO('model/intersection_model.pt')
# model.predict(source="sources/parking.webp", imgsz=736, conf=0.6, save=True)
# model.predict(source="sources/parking.mp4", imgsz=736, conf=0.6, save=True)

# model2 = YOLO('model/large_model.pt')
# model2.predict(source="sources/parking.webp", imgsz=736, conf=0.6, save=True)
# model2.predict(source="sources/parking.mp4", imgsz=736, conf=0.6, save=True)

# model3 = YOLO('model/small_model.pt')
# model3.predict(source="sources/parking.webp", imgsz=736, conf=0.6, save=True)
# model3.predict(source="sources/parking.mp4", imgsz=736, conf=0.6, save=True)

model4 = YOLO('model/tuned_model.pt')
model4.predict(source="sources/parking.webp", imgsz=1280, conf=0.6, save=True)
model4.predict(source="sources/parking.mp4", imgsz=1280, conf=0.6, save=True)
