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

from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image as PILImage
from scipy.spatial import Delaunay
import torch.nn.functional as F
from flask import make_response

import math




app = Flask(__name__)
CORS(
    app,
    resources={r"/api/*": {
        "origins": "http://localhost:3000",
        "expose_headers": ["X-Surface-Area"],  # Explicitly expose your custom header
        "supports_credentials": True
    }}
)


material_model = None
material_class_names = ['asphalt', 'concrete', 'glass'] 

midas = None
midas_transform = None

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

def estimate_surface_area(depth, mask, reference_length_px=None, reference_length_real=None, fx=1.0, fy=1.0, cx=None, cy=None):
    """
    Estimate 3D surface area from a depth map and mask with optional real-world scaling.
    
    Args:
        depth: Depth map from MiDaS
        mask: Binary mask of target surface
        reference_length_px: Length of known object in pixels
        reference_length_real: Real-world length of known object (same units as desired output)
        fx, fy: Focal lengths in pixels. If unknown, will be estimated.
        cx, cy: Principal point (image center if None)
    """
    ys, xs = np.where(mask > 127)
    zs = depth[ys, xs]

    if len(zs) < 3:
        return 0.0

    h, w = depth.shape
    if cx is None: cx = w / 2
    if cy is None: cy = h / 2
    
    # Estimate focal length if not provided
    if fx == 1.0 or fy == 1.0:
        fx = fy = max(w, h)  # Reasonable default for standard cameras
    
    # Apply real-world scaling if reference is provided
    if reference_length_px and reference_length_real:
        scale_factor = reference_length_real / reference_length_px
        zs = zs * scale_factor
        fx = fx * scale_factor
        fy = fy * scale_factor

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs
    points = np.stack([X, Y, Z], axis=1)

    tri = Delaunay(np.stack([xs, ys], axis=1))
    triangles = points[tri.simplices]

    def triangle_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    areas = [triangle_area(t[0], t[1], t[2]) for t in triangles]
    return float(np.sum(areas))


def initialize_midas():
    """Initialize MiDaS depth estimation model"""
    global midas, midas_transform
    
    print("Loading MiDaS model...")
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    midas.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    midas_transform = midas_transforms.dpt_transform
    print("MiDaS model loaded successfully")



def initialize_material_model():
    """Initialize the material classification model"""
    global material_model
    
    print("Loading material classification model...")
    model_path = "weights/material_classifier_effnet_b3.pth" 
    
    # Initialize model architecture
    material_model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=3)
    
    # Load state dict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    material_model.load_state_dict(torch.load(model_path, map_location=device))
    material_model = material_model.to(device)
    material_model.eval()
    print("Material classification model loaded successfully")

# Add this transform definition (put it near your other configuration)
material_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def initialize_models():
    """Initialize all models"""
    global model, sam, predictor, material_model, midas
    
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
    
    # Load material classification model
    initialize_material_model()
    
    # Load MiDaS model
    initialize_midas()
    
    print("All models loaded successfully")

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

@app.route('/api/classify/material', methods=['POST'])
def classify_material():
    """API endpoint for material classification"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Ensure model is loaded
    if material_model is None:
        initialize_material_model()
    
    try:
        # Read image
        image_bytes = file.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        img_tensor = material_transform(image_rgb).unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img_tensor = img_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            outputs = material_model(img_tensor)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get results
        pred_class = material_class_names[preds[0]]
        confidence = probs[0][preds[0]].item()
        
        return jsonify({
            'material': pred_class,
            'confidence': confidence,
            'class_id': int(preds[0]),
            'probabilities': {
                cls: float(probs[0][i].item())
                for i, cls in enumerate(material_class_names)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/depth/masked', methods=['POST'])
def get_masked_depth():
    """Endpoint that returns masked depth map and surface area"""
    if 'image' not in request.files or 'mask' not in request.files:
        return jsonify({'error': 'Both image and mask files are required'}), 400

    # Get optional reference parameters
    ref_px = request.form.get('reference_px', type=float)
    ref_real = request.form.get('reference_real', type=float)
    fx = request.form.get('fx', 1.0, type=float)
    fy = request.form.get('fy', 1.0, type=float)

    try:
        # Read and process files
        img = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imdecode(np.frombuffer(request.files['mask'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Initialize model if needed
        if midas is None:
            initialize_midas()

        # Estimate depth
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tensor = midas_transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_tensor)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy()
        masked_depth = np.where(mask > 127, depth, 0)

        # Estimate surface area with optional scaling
        surface_area = estimate_surface_area(
            depth, mask,
            reference_length_px=ref_px,
            reference_length_real=ref_real,
            fx=fx, fy=fy
        )

        # Prepare response
        depth_normalized = cv2.normalize(masked_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        _, buffer = cv2.imencode('.png', depth_normalized)

            # Create response
        response = make_response(send_file(
            io.BytesIO(buffer),
            mimetype='image/png',
            as_attachment=True,
            download_name='depth_map.png'
        ))
        
        # Set headers
        response.headers['X-Surface-Area'] = str(surface_area)
        response.headers['Access-Control-Expose-Headers'] = 'X-Surface-Area'
        
        # Debug output
        print(f"[DEBUG] Headers being sent: {response.headers}")
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    app.run(debug=True, port=5000)