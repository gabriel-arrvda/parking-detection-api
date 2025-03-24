from ultralytics import YOLO
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("../parking-lot-credentials.json")
app = firebase_admin.initialize_app(cred)
db = firestore.client()

doc_ref = db.collection("users").document("alovelace")

# model = YOLO('model/tuned_model.pt')
# model.predict(source="sources/parking.webp", imgsz=1280, conf=0.6, save=True)
# model.predict(source="sources/parking.mp4", imgsz=1280, conf=0.6, save=True)
