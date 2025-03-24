from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime

def init_firebase():
    cred = credentials.Certificate("../parking-lot-credentials.json")
    app = firebase_admin.initialize_app(cred)
    return firestore.client()

def process_results(results):
    detections = []
    for result in results:
        for box in result.boxes:
            detection = {
                'class_id': int(box.cls),
                'class_name': result.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist(),
                'timestamp': datetime.now()
            }

            detections.append(detection)

    return detections

db = init_firebase()

collection_ref = db.collection("results")

model = YOLO('model/tuned_model.pt')
# model.predict(source="sources/parking.webp", imgsz=1280, conf=0.6, save=True)
results = model.predict(source="sources/parking.webp", imgsz=1280, conf=0.6)

detections = process_results(results)

for result in detections:
    doc_ref = collection_ref.document()
    doc_ref.set(result)
    print(doc_ref.id)