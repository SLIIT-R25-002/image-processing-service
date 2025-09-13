# ai_logic.py
# Central AI logic for:
#  - Part 1: YOLO + SAM + CLIP material analysis (returns base64 masks per material)
#  - Part 2: DepthPro-based surface area estimation from an input image + mask

import os
import io
import base64
import json
import cv2
import numpy as np
from PIL import Image
from collections import Counter

import torch
from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor
import clip
from skimage import morphology

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import open3d as o3d

# =========================
# Configuration / Globals
# =========================

# Model paths can be overridden via environment variables
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "/weights/best.pt")
SAM_MODEL_PATH = os.getenv("SAM_MODEL_PATH", "/weights/mobile_sam.pt")
DEPTH_MODEL_ID = os.getenv("DEPTH_MODEL_ID", "apple/DepthPro-hf")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cityscapes-ish class mapping (as used in your prior code)
CLASS_NAMES = {
    0: "building",
    1: "road",
    2: "sidewalk",
    3: "vegetation",
    4: "wall",
}

# Materials palette (for visualization if needed)
COLOR_PALETTE = {
    "brick": [1.0, 0.3, 0.3],
    "concrete": [0.7, 0.7, 0.7],
    "glass": [0.3, 0.8, 1.0],
    "metal": [0.8, 0.8, 0.8],
    "asphalt": [0.2, 0.2, 0.2],
    "pavers": [0.9, 0.5, 0.2],
}

# Prompts for CLIP scoring (same spirit as your notebook)
CLASS_MATERIAL_PROMPTS = {
    0: {  # building
        "materials": ["brick", "concrete", "glass"],
        "templates": {
            "brick": [
                "close-up of red brick texture with visible mortar lines",
                "brick wall with staggered bond pattern, no windows",
                "textured brick facade under sunlight",
                "weathered brick with small cracks and dirt",
            ],
            "concrete": [
                "smooth poured concrete with formwork marks",
                "gray concrete wall without repeating units",
                "unfinished concrete (béton brut) surface",
                "concrete plinth with coarse texture",
            ],
            "glass": [
                "a glass window reflecting a blue sky and white clouds",
                "the reflection of the sky on a modern glass window",
                "a dark window pane on a brick building",
                "a grid of glass windows on a building facade",
                "a dark tinted glass window on a building",
                "a window reflecting green trees and a clear sky",
                "a clean glass panel showing the interior of a room",
                "a dusty or dirty window pane",
                "a skyscraper facade made of reflective blue glass",
                "reflective glass window on urban building",
                "transparent glass panel with metal frame",
                "glass facade showing sky reflection",
            ]
        },
    },
    1: {  # road
        "materials": ["asphalt", "concrete"],
        "templates": {
            "asphalt": [
                "black asphalt road with tire marks",
                "granular asphalt texture with small stones",
                "dark bitumen pavement, slightly wet",
            ],
            "concrete": [
                "light gray concrete road with panel joints",
                "paved concrete highway with expansion gaps",
            ],
        },
    },
    2: {  # sidewalk
        "materials": ["concrete", "brick", "pavers"],
        "templates": {
            "concrete": [
                "plain concrete sidewalk, cracked surface",
                "smooth concrete path with no texture",
            ],
            "brick": [
                "red brick pavers on sidewalk",
                "interlocking brick tiles in herringbone pattern",
            ],
            "pavers": [
                "stone pavers with sand-filled joints",
                "rectangular paving stones on pedestrian path",
            ],
        },
    },
    4: {  # wall
        "materials": ["brick", "concrete", "metal", "glass"],
        "templates": {
            "brick": ["brick wall with mortar lines"],
            "concrete": ["concrete retaining wall"],
            "metal": ["metal fence"],
            "glass": ["a glass barrier reflecting the sky", "a dark glass panel on a wall"],
        },
    },
}

# Lazy singletons (loaded once per worker)
_yolo = None
_sam = None
_sam_pred = None
_clip_model = None
_clip_preprocess = None
_depth_processor = None
_depth_model = None

print("🚀 Initializing AI backends... (models will lazy-load on first use)")

# =========================
# Utilities
# =========================

def _ensure_yolo():
    global _yolo
    if _yolo is None:
        if not os.path.exists(YOLO_MODEL_PATH):
            raise FileNotFoundError(f"YOLO model file not found: {YOLO_MODEL_PATH}")
        print("📦 Loading YOLO model...")
        _yolo = YOLO(YOLO_MODEL_PATH)
    return _yolo

def _ensure_sam():
    global _sam, _sam_pred
    if _sam_pred is None:
        if not os.path.exists(SAM_MODEL_PATH):
            raise FileNotFoundError(f"SAM checkpoint not found: {SAM_MODEL_PATH}")
        print("🧠 Loading Mobile SAM...")
        _sam = sam_model_registry["vit_t"](checkpoint=SAM_MODEL_PATH)
        _sam.to(DEVICE)
        _sam_pred = SamPredictor(_sam)
    return _sam_pred

def _ensure_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        print("🔤 Loading CLIP ViT-B/32...")
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=CLIP_DEVICE)
    return _clip_model, _clip_preprocess

def _ensure_depth():
    global _depth_processor, _depth_model
    if _depth_model is None:
        print("🌊 Loading DepthPro model...")
        _depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        _depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE)
        _depth_model.eval()
    return _depth_processor, _depth_model

def _read_image_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _mask_to_base64_png(mask_bool: np.ndarray) -> str:
    """
    Convert a boolean mask (H, W) to base64-encoded PNG (grayscale 0/255).
    """
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    pil = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _base64_png_to_mask(b64: str) -> np.ndarray:
    """
    Convert base64-encoded PNG (grayscale) back to boolean mask.
    """
    data = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(data)).convert("L")
    arr = np.array(pil)
    return (arr > 128)

# =========================
# CLIP material classification (enhanced)
# =========================

def _prepare_clip_prompts_for_class(cls_id: int):
    """
    Build a tokenized prompt list and material keyword map for a given class id.
    """
    _clip_model, _clip_preprocess = _ensure_clip()
    if cls_id not in CLASS_MATERIAL_PROMPTS:
        templates = [f"a photo of {CLASS_NAMES.get(cls_id, 'object')}"]
        material_keywords = {CLASS_NAMES.get(cls_id, 'object'): CLASS_NAMES.get(cls_id, 'object')}
    else:
        all_templates = []
        material_keywords = {}
        for mat, prompts in CLASS_MATERIAL_PROMPTS[cls_id]["templates"].items():
            for p in prompts:
                all_templates.append(p)
                # map keyword -> canonical material
                material_keywords[mat.lower()] = mat
        templates = all_templates

    text_tokens = clip.tokenize(templates).to(CLIP_DEVICE)
    return templates, material_keywords, text_tokens, _clip_preprocess, _clip_model

def _classify_material_enhanced(pil_img: Image.Image, templates, material_keywords, text_tokens, clip_preprocess, clip_model):
    """
    Returns (best_material, score) based on a hybrid heuristic + CLIP.
    """
    np_img = np.array(pil_img)
    if np_img.size == 0:
        return "unknown", 0.0

    h, w = np_img.shape[:2]
    if h == 0 or w == 0:
        return "unknown", 0.0

    # 1) CLIP score (50%)
    image_input = clip_preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()[0]

    clip_scores = {}
    for idx, prob in enumerate(probs):
        prompt = templates[idx].lower()
        for keyword, material in material_keywords.items():
            if keyword in prompt:
                clip_scores[material] = max(clip_scores.get(material, 0.0), float(prob))
                break

    # 2) Heuristics (color/texture/reflection) total 50% (0.2 + 0.2 + 0.1 with a small 0.0 slack)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    mean_color = np_img.mean(axis=(0, 1))  # RGB
    texture_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    brightness = gray.mean()

    color_scores = {}
    texture_scores = {}
    reflection_scores = {}

    # Brick-like reddish/brownish tendency
    if 100 < mean_color[0] < 200 and mean_color[1] < mean_color[0] * 0.7 and mean_color[2] < mean_color[0] * 0.7:
        color_scores["brick"] = 1.0
    # Glass tends to skew blue/neutral with reflections/low texture
    if mean_color[2] > mean_color[0] and mean_color[2] > mean_color[1]:
        color_scores["glass"] = max(color_scores.get("glass", 0.0), 1.0)

    if texture_var > 100:
        texture_scores["brick"] = 1.0
        texture_scores["pavers"] = 1.0
    elif texture_var < 35:
        texture_scores["glass"] = 1.0
        texture_scores["concrete"] = max(texture_scores.get("concrete", 0.0), 0.8)

    if brightness > 180 or contrast > 60:
        reflection_scores["glass"] = max(reflection_scores.get("glass", 0.0), 1.0)
    if np.std(np_img) < 40:
        reflection_scores["glass"] = max(reflection_scores.get("glass", 0.0), 0.9)

    final_scores = Counter()
    all_materials = set(clip_scores) | set(color_scores) | set(texture_scores) | set(reflection_scores)
    for mat in all_materials:
        score = (
            clip_scores.get(mat, 0.0) * 0.5
            + color_scores.get(mat, 0.0) * 0.2
            + texture_scores.get(mat, 0.0) * 0.2
            + reflection_scores.get(mat, 0.0) * 0.1
        )
        final_scores[mat] = score

    if not final_scores:
        return "unknown", 0.0

    best_mat, best_score = final_scores.most_common(1)[0]
    return best_mat, float(best_score)

# =========================
# =========================
# Part 1: Analysis Pipeline (IMPROVED ACCURACY VERSION)
# =========================

def run_analysis_pipeline(image_path: str):
    """
    Detect -> Segment -> Hierarchically segment -> Classify materials.
    Returns a JSON-serializable dict with base64 masks per (class -> material).
    This version is more accurate but slower than the simpler one.
    """
    print(f"🧠 Analyzing image with high-accuracy method: {image_path}")

    # Load models
    yolo = _ensure_yolo()
    sam_pred = _ensure_sam()

    # Read image
    image_rgb = _read_image_rgb(image_path)
    H, W = image_rgb.shape[:2]

    # YOLO detection
    print("🔍 YOLO inference...")
    results = yolo(image_path)

    bboxes, class_ids = [], []
    for r in results:
        if getattr(r, "boxes", None) is not None:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                bboxes.append(box.detach().cpu().numpy())
                class_ids.append(int(cls.detach().cpu().numpy()))

    if not bboxes:
        return {"message": "No objects detected", "masks": {}}

    # SAM segmentation for each bbox
    print("✂️ SAM segmentation for initial instances...")
    sam_pred.set_image(image_rgb)
    instance_masks = []
    for box in bboxes:
        box = np.array(box)
        try:
            mask, _, _ = sam_pred.predict(box=box, multimask_output=False)
            mask = mask.squeeze().astype(bool)
            mask = morphology.remove_small_objects(mask, min_size=500)
            mask = morphology.remove_small_holes(mask, area_threshold=500)
        except Exception as e:
            print(f"⚠️ SAM failed on {box}: {e}")
            mask = np.zeros((H, W), dtype=bool)
        instance_masks.append(mask)

    # This will hold the final aggregated masks for all materials across all instances
    # e.g., {"building": {"glass": <numpy_mask>, "brick": <numpy_mask>}}
    final_material_masks = {}

    print("🧪 Starting hierarchical material classification...")
    # Loop through each detected instance (e.g., each building)
    for obj_idx, (instance_mask, cls_id) in enumerate(zip(instance_masks, class_ids)):
        if cls_id not in CLASS_NAMES or cls_id not in CLASS_MATERIAL_PROMPTS:
            continue
        
        class_name = CLASS_NAMES[cls_id]
        print(f"\n🧩 Processing {class_name} instance #{obj_idx + 1}...")

        # --- Hierarchical Segmentation Logic Starts Here ---

        # 1. Crop the image and mask to the bounding box of the instance
        ys, xs = np.where(instance_mask)
        if ys.size == 0:
            continue
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        
        cropped_img = image_rgb[y1:y2, x1:x2]
        cropped_mask = instance_mask[y1:y2, x1:x2]

        if cropped_img.size == 0:
            continue

        # 2. Set the SAM predictor to the *cropped* image
        sam_pred.set_image(cropped_img)
        
        # 3. Generate a grid of points inside the cropped mask to find sub-parts
        h, w = cropped_mask.shape
        step_h, step_w = max(1, h // 4), max(1, w // 4)
        points = [
            [j * step_w, i * step_h]
            for i in range(1, 4)
            for j in range(1, 4)
            if (i * step_h < h and j * step_w < w and cropped_mask[i * step_h, j * step_w])
        ]

        if not points:
            continue

        # 4. Get masks for all sub-parts from the points
        part_masks_local = [] # Masks in local (cropped) coordinates
        for pt in points:
            try:
                mask, _, _ = sam_pred.predict(
                    point_coords=np.array([pt]),
                    point_labels=np.array([1]),
                    multimask_output=False,
                )
                part_masks_local.append(mask.squeeze())
            except Exception:
                continue

        # 5. De-duplicate the sub-part masks using IoU
        unique_parts_local = []
        for pm in part_masks_local:
            if pm.sum() < 100: continue # Ignore tiny parts
            is_unique = True
            for exist in unique_parts_local:
                iou = (pm & exist).sum() / (pm | exist).sum()
                if iou > 0.7:
                    is_unique = False
                    break
            if is_unique:
                unique_parts_local.append(pm)
        
        print(f"  ✅ Found {len(unique_parts_local)} unique material segments inside.")

        # 6. Classify each unique sub-part
        templates, material_keywords, text_tokens, clip_preprocess, clip_model = _prepare_clip_prompts_for_class(cls_id)
        
        for part_mask_local in unique_parts_local:
            # Crop the sub-part for classification
            pys, pxs = np.where(part_mask_local)
            if pys.size == 0: continue
            py1, py2 = pys.min(), pys.max()
            px1, px2 = pxs.min(), pxs.max()
            
            part_crop_img = cropped_img[py1:py2, px1:px2]
            pil_image = Image.fromarray(part_crop_img)

            material, score = _classify_material_enhanced(
                pil_image, templates, material_keywords, text_tokens, clip_preprocess, clip_model
            )
            
            if material == "unknown":
                continue

            # 7. Add the classified part-mask to the final global mask dictionary
            if class_name not in final_material_masks:
                final_material_masks[class_name] = {}
            if material not in final_material_masks[class_name]:
                final_material_masks[class_name][material] = np.zeros((H, W), dtype=bool)
            
            # Create a full-size mask for this part and add it
            full_part_mask = np.zeros((H, W), dtype=bool)
            # Place the local mask at the correct global offset (y1, x1)
            full_part_mask[y1:y2, x1:x2][part_mask_local] = True
            
            # Combine with the main material mask
            final_material_masks[class_name][material] |= full_part_mask

    # --- Hierarchical Segmentation Logic Ends Here ---

    # Convert final aggregated masks to base64 PNG strings
    print("\n🧾 Encoding final aggregated masks to base64...")
    encoded = {}
    for cls_name, mats in final_material_masks.items():
        encoded[cls_name] = {}
        for mat_name, mat_mask in mats.items():
            if np.count_nonzero(mat_mask) == 0:
                continue
            encoded[cls_name][mat_name] = {
                "mask": _mask_to_base64_png(mat_mask)
            }

    result = {
        "detected_classes": [
            {"id": cid, "name": CLASS_NAMES.get(cid, f"class_{cid}")}
            for cid in sorted(set(class_ids))
        ],
        "masks": encoded,
    }
    print("✅ High-accuracy analysis complete.")
    return result

# =========================
# Part 2: Area Calculation
# =========================

def _create_point_cloud(depth_map, fx, fy, cx, cy):
    h, w = depth_map.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = ~np.isnan(depth_map)
    d = depth_map[valid]
    if d.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    u_valid, v_valid = u[valid], v[valid]
    x = (u_valid - cx) * d / fx
    y = (v_valid - cy) * d / fy
    z = d
    return np.vstack((x, y, z)).T

def run_area_calculation(image_path: str, mask_base64: str, real_distance: float) -> float:
    """
    - image_path: path to original RGB image
    - mask_base64: base64 PNG of a grayscale mask (255 inside building)
    - real_distance: user-estimated real distance to the closest visible part of the building (meters)

    Returns: estimated surface area (square meters)
    """
    print(f"📐 Calculating area for image: {image_path}")
    # Load image
    rgb = _read_image_rgb(image_path)
    H, W = rgb.shape[:2]

    # Decode mask and resize to image size if needed
    mask_bool = _base64_png_to_mask(mask_base64)
    if mask_bool.shape != (H, W):
        mask_bool = cv2.resize(mask_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

    # DepthPro inference
    depth_processor, depth_model = _ensure_depth()
    pil_img = Image.fromarray(rgb)
    with torch.no_grad():
        inputs = depth_processor(images=pil_img, return_tensors="pt").to(DEVICE)
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth  # [B, H', W'] relative depth
        # Resize to image size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    depth_raw = prediction.detach().cpu().numpy()

    # Apply mask
    depth_masked = np.where(mask_bool, depth_raw, np.nan)

    # Scaling with robust 1st percentile
    positive_vals = depth_masked[depth_masked > 0]
    if positive_vals.size == 0:
        print("❌ No positive depth values in mask. Returning 0.")
        return 0.0

    robust_min = np.nanpercentile(positive_vals, 1)
    if robust_min <= 0 or not np.isfinite(robust_min):
        print("❌ Invalid robust minimum. Returning 0.")
        return 0.0

    if real_distance is not None and real_distance > 0:
        scale = float(real_distance) / float(robust_min)
        depth_scaled = depth_masked * scale
        depth_scaled[depth_scaled <= 0] = np.nan
    else:
        # Unscaled (relative units) — area will not be in meters²
        depth_scaled = depth_masked

    # Point cloud
    focal = float(W)  # crude guess; for better accuracy, pass actual intrinsics
    cx, cy = W / 2.0, H / 2.0
    pts = _create_point_cloud(depth_scaled, focal, focal, cx, cy)
    if pts.shape[0] == 0:
        print("❌ No points created. Returning 0.")
        return 0.0

    # Mesh (Poisson) & area
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        densities = np.asarray(densities)
        keep = densities >= np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(~keep)
        area = float(mesh.get_surface_area())
        print(f"✅ Estimated Surface Area: {area:.2f} m²")
        return area
    except Exception as e:
        print(f"❌ Mesh reconstruction error: {e}")
        return 0.0
