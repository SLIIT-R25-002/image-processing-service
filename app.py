import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
from skimage import morphology
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}}, supports_credentials=True)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model paths (adjust these to your local paths)
YOLO_MODEL_PATH = "weights/best.pt"  # Update this path
SAM_MODEL_PATH = "weights/sam_vit_b_01ec64.pth"  # Update this path

# Initialize models (None initially, loaded on first request)
model = None
sam = None
predictor = None

# Class definitions
class_colors = {
    0: [0.8, 0.1, 0.1, 0.7],  # Deeper red for buildings
    1: [0.1, 0.1, 0.8, 0.7],   # Deeper blue for roads
    2: [0.1, 0.8, 0.8, 0.7],   # Brighter cyan for sidewalks
    3: [0.1, 0.8, 0.1, 0.7],   # Brighter green for vegetation
    4: [0.5, 0.5, 0.5, 0.7]    # Neutral gray for walls
}

class_names = {
    0: 'building',
    1: 'road',
    2: 'sidewalk',
    3: 'vegetation',
    4: 'wall'
}

def initialize_models():
    """Initialize YOLO and SAM models"""
    global model, sam, predictor
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL_PATH)
    
    # Load SAM model
    print("Loading SAM model...")
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=SAM_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam.to(device)
    predictor = SamPredictor(sam)
    print("Models loaded successfully")

def process_image(image):
    """Process image with YOLO and SAM models"""
    global model, predictor
    
    # Ensure models are loaded
    if model is None or predictor is None:
        initialize_models()
    
    # Process with YOLO
    results = model(image)
    
    # Extract bounding boxes & class ids
    bboxes, class_ids = [], []
    for r in results:
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box, class_id in zip(r.boxes.xyxy, r.boxes.cls):
                bboxes.append(box.cpu().numpy())
                class_ids.append(int(class_id.cpu().numpy()))

    if not bboxes:
        return None, None
    
    # Process with SAM
    predictor.set_image(image)
    
    # Process each bounding box and extract masks
    all_masks = []
    all_class_ids = []
    
    for box, class_id in zip(bboxes, class_ids):
        box = np.array(box)
        
        if class_id == 0:  # Special handling for buildings
            masks_multi, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=True
            )
            
            for mask in masks_multi:
                mask = mask.squeeze()
                mask = morphology.remove_small_objects(mask, min_size=100)
                mask = morphology.remove_small_holes(mask, area_threshold=100)
                
                if np.sum(mask) >= 200:
                    all_masks.append(mask)
                    all_class_ids.append(class_id)
        else:
            mask, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False
            )
            mask = mask.squeeze()
            mask = morphology.remove_small_objects(mask, min_size=100)
            mask = morphology.remove_small_holes(mask, area_threshold=100)
            all_masks.append(mask)
            all_class_ids.append(class_id)
    
    return all_masks, all_class_ids


@app.route('/api/segment/all', methods=['POST'])
def segment_all():
    """API endpoint for full segmentation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read image
    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    all_masks, all_class_ids = process_image(image)
    if all_masks is None:
        return jsonify({'error': 'No objects detected'}), 400
    
   # Create visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(image)

    for mask, class_id in zip(all_masks, all_class_ids):
        color = class_colors.get(class_id, [1, 1, 1, 0.5])  # RGBA
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask > 0] = color
        plt.imshow(overlay)

    plt.axis("off")

    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    
    return send_file(buf, mimetype='image/png')

@app.route('/api/segment/class', methods=['POST'])
def segment_class():
    """API endpoint for class-specific segmentation"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    class_input = request.form.get('class_id', '')
    
    if not class_input:
        return jsonify({'error': 'No class_id provided'}), 400
    
    # Convert class input to ID
    try:
        class_id = int(class_input)
    except ValueError:
        # Try to match by name
        class_id = next((k for k, v in class_names.items() if v == class_input.lower()), None)
        if class_id is None:
            return jsonify({'error': f'Invalid class identifier: {class_input}'}), 400
    
    # Read image
    image_bytes = file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process image
    all_masks, all_class_ids = process_image(image)
    if all_masks is None:
        return jsonify({'error': 'No objects detected'}), 400
    
    # Filter masks for the requested class
    class_masks = [mask for mask, cid in zip(all_masks, all_class_ids) if cid == class_id]
    
    if not class_masks:
        return jsonify({'error': f'No masks found for class: {class_names.get(class_id, class_id)}'}), 404
    
    # Combine masks and create output
    combined_mask = np.zeros(image.shape[:2], dtype=bool)
    for mask in class_masks:
        combined_mask = np.logical_or(combined_mask, mask)
    
    output_image = np.zeros_like(image)
    output_image[combined_mask] = image[combined_mask]
    
    # Convert to PNG and return
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    if not is_success:
        return jsonify({'error': 'Failed to encode image'}), 500
    
    return send_file(
        io.BytesIO(buffer),
        mimetype='image/png'
    )

@app.route('/api/classes', methods=['GET'])
def list_classes():
    """API endpoint to list available classes"""
    return jsonify({
        'classes': [
            {'id': k, 'name': v}
            for k, v in class_names.items()
        ]
    })



if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    app.run(debug=True, port=5000)