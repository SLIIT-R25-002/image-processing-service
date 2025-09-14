import os
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import torch

from ai_logic import run_analysis_pipeline, run_area_calculation, _yolo, _sam, _clip_model, _depth_model, preload_all_models

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for device and models
device = None

def initialize_device():
    """Initialize and configure the global device for CUDA/CPU"""
    global device
    
    try:
        # Check CUDA availability and configure device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Log device information similar to SuperGlue backend
        logger.info(f'Running inference on device: {device}')
        logger.info(f'CUDA version: {torch.version.cuda}')
        logger.info(f'CUDA available: {torch.cuda.is_available()}')
        
        if torch.cuda.is_available():
            logger.info(f'GPU device name: {torch.cuda.get_device_name(0)}')
            logger.info(f'GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
            logger.info(f'GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
            
            # Clear any existing CUDA cache
            torch.cuda.empty_cache()
            logger.info('CUDA cache cleared')
        else:
            logger.info('No CUDA GPU available, running on CPU')
    
    except Exception as e:
        logger.error(f'Error initializing device: {e}')
        device = torch.device("cpu")
        logger.info('Falling back to CPU due to CUDA initialization error')
    
    return device

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR

# Initialize device on startup
device = initialize_device()

# Preload all AI models during startup to avoid delays during analysis
try:
    print("🤖 Starting AI model initialization...")
    preload_all_models()
    print("🎉 All AI models ready for inference!")
except Exception as e:
    logger.error(f"❌ Failed to preload AI models: {e}")
    logger.error("⚠️ Models will be loaded on first use (lazy loading)")

# ---------- Endpoint 1: Run analysis synchronously ----------
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

    result = run_analysis_pipeline(image_path)
    return jsonify({"status": "SUCCESS", "result": result}), 200

# ---------- Endpoint 2: Calculate area ----------
@app.route("/calculate_area", methods=["POST"])
def calculate_area_endpoint():
    logger.info("=== Starting calculate_area_endpoint ===")
    
    # Log request details
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Request content type: {request.content_type}")
    
    data = request.get_json(silent=True)
    logger.info(f"Received JSON data: {data}")
    
    if not data:
        logger.error("No JSON body received in request")
        return jsonify({"error": "Missing JSON body"}), 400

    required = {"image_filename", "mask_base64", "real_distance"}
    logger.info(f"Required fields: {required}")
    logger.info(f"Received fields: {set(data.keys()) if data else 'None'}")
    
    if not required.issubset(data):
        missing_keys = required - set(data.keys())
        logger.error(f"Missing required keys: {missing_keys}")
        return jsonify({"error": f"Missing keys. Required: {sorted(list(required))}"}), 400

    # Log received parameters
    logger.info(f"Image filename: {data['image_filename']}")
    logger.info(f"Real distance: {data['real_distance']}")
    logger.info(f"Mask base64 length: {len(data['mask_base64']) if data['mask_base64'] else 'None'}")

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(data["image_filename"]))
    logger.info(f"Constructed image path: {image_path}")
    logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    if not os.path.exists(image_path):
        logger.error(f"Image file not found at path: {image_path}")
        logger.info(f"Available files in upload folder: {os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else 'Upload folder does not exist'}")
        return jsonify({"error": "Image not found on server"}), 404

    logger.info(f"Image file exists, size: {os.path.getsize(image_path)} bytes")

    try:
        real_distance = float(data["real_distance"])
        logger.info(f"Successfully parsed real_distance: {real_distance}")
    except Exception as e:
        logger.error(f"Failed to parse real_distance '{data['real_distance']}': {e}")
        return jsonify({"error": "real_distance must be a number"}), 400

    logger.info("Starting area calculation...")
    try:
        area = run_area_calculation(
            image_path=image_path,
            mask_base64=data["mask_base64"],
            real_distance=real_distance,
        )
        logger.info(f"Area calculation completed successfully: {area} square meters")
    except Exception as e:
        logger.error(f"Error during area calculation: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Area calculation failed: {str(e)}"}), 500

    result = {"surface_area": area, "unit": "square_meters"}
    logger.info(f"Returning result: {result}")
    logger.info("=== calculate_area_endpoint completed successfully ===")
    
    return jsonify(result), 200

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint to verify the service is running"""
    global device
    
    # Ensure device is initialized
    if device is None:
        initialize_device()
    
    health_info = {
        "status": "healthy",
        "service": "image-processing-service",
        "message": "Service is running normally",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "models_loaded": {
            "yolo": _yolo is not None,
            "sam": _sam is not None,
            "clip": _clip_model is not None,
            "depth": _depth_model is not None
        }
    }
    
    # Add GPU info if CUDA is available
    if torch.cuda.is_available():
        health_info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "gpu_memory_cached_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        })
    
    return jsonify(health_info), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
