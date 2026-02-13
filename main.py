from typing import List
from nudenet import NudeDetector
from PIL import Image
import uuid
import os
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse


app = FastAPI(title="NudeNet Detection API")

# print("🔥🔥 NEW CODE IS RUNNING 🔥🔥")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "status": False,
            "message": str(exc),
        }
    )

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

@app.post("/image-detect")
async def detect_images(files: List[UploadFile] = File(...)):
    if not files:
        return {
            "status": False,
            "message": "No files uploaded",
            "results": []
        }
    
    for file in files:
        if not file.filename:
            return {
                "status": False,
                "message": "No image uploaded"
            }
    if not file.content_type or not file.content_type.startswith("image/"):
            return {
                "status": False,
                "message": "Only image files are allowed"
            }

    results = []

    for file in files:
        image_id = f"{uuid.uuid4()}.jpg"
        path = os.path.join(UPLOAD_DIR, image_id)

        try:
            image = Image.open(file.file).convert("RGB")
            image.save(path)

            detections = detector.detect(path)
            category = classify(detections)

            results.append({
                "id": image_id,
                "filename": file.filename,
                "status": True,
                "category": category,
                "detections": detections
            })

        except Exception as e:
            results.append({
                "status": False,
                "message": str(e)
            })

        finally:
            if os.path.exists(path):
                os.remove(path)

    return {
        "status": True,
        "count": len(results),
        "results": results
    }

def classify_video(detections):
    if not isinstance(detections, list):
        return "SAFE"

    male_genital = female_genital = anus = False
    breast = belly = buttocks = genital_covered = False

    for d in detections:
        if not isinstance(d, dict):
            continue

        label = d.get("class", "")
        score = float(d.get("score", 0))

        # --- NUDE ---
        if label == "MALE_GENITALIA_EXPOSED" and score >= 0.3:
            male_genital = True
        if label == "FEMALE_GENITALIA_EXPOSED" and score >= 0.4:
            female_genital = True
        if label == "ANUS_EXPOSED" and score >= 0.4:
            anus = True

        # --- SEXY SIGNALS ---
        if label in {"FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED"} and score >= 0.5:
            breast = True

        if label == "BELLY_EXPOSED" and score >= 0.5:
            belly = True

        if label in {"BUTTOCKS_EXPOSED", "BUTTOCKS_COVERED"} and score >= 0.45:
            buttocks = True

        if label == "FEMALE_GENITALIA_COVERED" and score >= 0.6:
            genital_covered = True

    # --- FINAL DECISION ---
    if male_genital or female_genital or anus:
        return "NUDE"

    # Strong sexy combinations
    if sum([breast, belly, buttocks]) >= 2:
        return "SEXY"

    # Bikini / beach rule ⭐
    if belly and (breast or genital_covered):
        return "SEXY"

    return "SAFE"

def detect_video(video_path, frame_rate=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 1

    frame_count = 0
    frames_analyzed = 0
    video_category = "SAFE"

    label_scores = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % (fps * frame_rate) == 0:
            frames_analyzed += 1

            temp_path = os.path.join(UPLOAD_DIR, "temp_frame.jpg")
            cv2.imwrite(temp_path, frame)

            detections = detector.detect(temp_path)

            # aggregate scores
            for d in detections:
                label = d["class"]
                score = float(d["score"])
                label_scores[label] = max(label_scores.get(label, 0), score)

            category = classify_video(detections)

            if category == "NUDE":
                video_category = "NUDE"
                break
            elif category == "SEXY":
                video_category = "SEXY"

        frame_count += 1

    cap.release()

    temp_frame = os.path.join(UPLOAD_DIR, "temp_frame.jpg")
    if os.path.exists(temp_frame):
        os.remove(temp_frame)

    return {
        "category": video_category,
        "frames_analyzed": frames_analyzed,
        "scores": label_scores
    }

# --- Video detection endpoint ---
@app.post("/video-detect")
async def detect_video_endpoint(files: List[UploadFile] = File(...)):

    if not files:
        return {
            "status": False,
            "message": "No video uploaded"
        }

    for file in files:
        # 🚫 no file selected
        if not file.filename:
            return {
                "status": False,
                "message": "No video uploaded"
            }

        # 🚫 wrong extension
        if not file.filename.lower().endswith(".mp4"):
            return {
                "status": False,
                "message": "Only MP4 videos are allowed"
            }

        # 🚫 wrong mime type
        if file.content_type != "video/mp4":
            return {
                "status": False,
                "message": "Invalid video content type"
            }

    results = []

    for upload in files:
        video_id = f"{uuid.uuid4()}.mp4"
        path = os.path.join(UPLOAD_DIR, video_id)

        try:
            with open(path, "wb") as f:
                f.write(await upload.read())

            result = detect_video(path)

            results.append({
                "id": video_id,
                "filename": upload.filename,
                "category": result["category"],
                "frames_analyzed": result["frames_analyzed"],
                "scores": result["scores"]
            })

        except Exception:
            return {
                "status": False,
                "message": "Invalid or corrupted video file"
            }

        finally:
            if os.path.exists(path):
                os.remove(path)

    return {
        "status": True,
        "count": len(results),
        "results": results
    }


