# app.py
# Flask API with:
#  - POST /analyze -> accepts an image, starts Celery task, returns task_id
#  - GET  /status/<task_id> -> returns status/result of analysis
#  - POST /calculate_area -> calculates surface area using a base64 mask + real distance

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from celery.result import AsyncResult

from task import analyze_image_task
from ai_logic import run_area_calculation

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ---------- Endpoint 1: Start analysis (async) ----------
@app.route("/analyze", methods=["POST"])
def analyze_image_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(f.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(image_path)

    # Kick to Celery
    task = analyze_image_task.delay(image_path)
    return jsonify({"task_id": task.id, "image_filename": filename}), 202

# ---------- Endpoint 2: Check task status ----------
@app.route("/status/<task_id>", methods=["GET"])
def get_task_status(task_id):
    ar = AsyncResult(task_id, app=analyze_image_task.app)
    if ar.successful():
        try:
            data = ar.get()
        except Exception as e:
            return jsonify({"status": "FAILURE", "error": str(e)}), 500
        return jsonify({"status": "SUCCESS", "result": data}), 200
    elif ar.failed():
        return jsonify({"status": "FAILURE", "error": str(ar.info)}), 500
    else:
        return jsonify({"status": str(ar.status)}), 202  # PENDING / STARTED / RETRY

# ---------- Endpoint 3: Calculate area (sync) ----------
@app.route("/calculate_area", methods=["POST"])
def calculate_area_endpoint():
    """
    JSON body:
    {
      "image_filename": "your_uploaded.jpg",
      "mask_base64": "iVBORw0KGgoAAA...",  # base64 PNG of grayscale mask (255=inside)
      "real_distance": 12.5  # meters to nearest part of the building
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    required = {"image_filename", "mask_base64", "real_distance"}
    if not required.issubset(data):
        return jsonify({"error": f"Missing keys. Required: {sorted(list(required))}"}), 400

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(data["image_filename"]))
    if not os.path.exists(image_path):
        return jsonify({"error": "Image not found on server"}), 404

    try:
        real_distance = float(data["real_distance"])
    except Exception:
        return jsonify({"error": "real_distance must be a number"}), 400

    area = run_area_calculation(
        image_path=image_path,
        mask_base64=data["mask_base64"],
        real_distance=real_distance,
    )

    return jsonify({"surface_area": area, "unit": "square_meters"}), 200

if __name__ == "__main__":
    # Expose on all interfaces for dev; disable debug in production
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
