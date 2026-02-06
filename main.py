from fastapi import FastAPI, UploadFile, File
from typing import List
from nudenet import NudeDetector
from PIL import Image
import uuid
import os

app = FastAPI(title="NudeNet Detection API")

detector = NudeDetector()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def classify(detections):
    sexy = False

    male_genital = False
    female_genital = False
    anus = False
    buttocks = False

    for d in detections:
        label = d.get("class", "")
        score = d.get("score", 0)

        if label == "MALE_GENITALIA_EXPOSED" and score >= 0.3:
            male_genital = True

        if label == "FEMALE_GENITALIA_EXPOSED" and score >= 0.5:
            female_genital = True

        if label == "ANUS_EXPOSED" and score >= 0.5:
            anus = True

        if label == "BUTTOCKS_EXPOSED" and score >= 0.5:
            buttocks = True

        if label in {
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_BREAST_COVERED"
        } and score >= 0.6:
            sexy = True

    if male_genital or female_genital or anus:
        return "NUDE"

    if buttocks and (male_genital or anus):
        return "NUDE"

    if sexy:
        return "SEXY"

    return "SAFE"

@app.get("/")
def health():
    return {"status": "API is running"}

@app.post("/detect")
async def detect_images(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        image_id = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_DIR, image_id)

        image = Image.open(file.file).convert("RGB")
        image.save(path)

        detections = detector.detect(path)
        category = classify(detections)

        results.append({
            "id": image_id,
            "category": category,
            "detections": detections
        })

    return {
        "count": len(results),
        "results": results
    }
