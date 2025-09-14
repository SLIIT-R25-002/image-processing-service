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

from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
# from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import open3d as o3d

# =========================
# Configuration / Globals
# =========================

YOLO_MODEL_PATH = "/app/weights/best.pt"
SAM_MODEL_PATH = "/app/weights/mobile_sam.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = {
    0: "building", 1: "road", 2: "sidewalk", 3: "vegetation", 4: "wall",
}

# ❗ Material promts
CLASS_MATERIAL_PROMPTS = {
    0: {  # building
        "materials": ["brick", "concrete", "glass"],
        "templates": {
            "brick": [
                "close-up of red brick texture with visible mortar lines",
                "brick wall with staggered bond pattern, no windows",
                "textured brick facade under sunlight",
                "weathered brick with small cracks and dirt"
            ],
            "concrete": [
                "smooth poured concrete with formwork marks",
                "gray concrete wall without repeating units",
                "unfinished concrete (béton brut) surface",
                "concrete plinth with coarse texture"
            ],
            "glass": [
                "reflective glass window on urban building",
                "transparent glass panel with metal frame",
                "glass facade showing sky reflection",
                "modern office window with grid pattern"
            ]
        }
    },
    1: {  # road
        "materials": ["asphalt", "concrete"],
        "templates": {
            "asphalt": [
                "black asphalt road with tire marks",
                "granular asphalt texture with small stones",
                "dark bitumen pavement, slightly wet"
            ],
            "concrete": [
                "light gray concrete road with panel joints",
                "paved concrete highway with expansion gaps"
            ]
        }
    },
    2: {  # sidewalk
        "materials": ["concrete", "brick", "pavers"],
        "templates": {
            "concrete": [
                "plain concrete sidewalk, cracked surface",
                "smooth concrete path with no texture"
            ],
            "brick": [
                "red brick pavers on sidewalk",
                "interlocking brick tiles in herringbone pattern"
            ],
            "pavers": [
                "stone pavers with sand-filled joints",
                "rectangular paving stones on pedestrian path"
            ]
        }
    },
    4: {  # wall
        "materials": ["brick", "concrete", "metal", "glass"],
        "templates": {
            "brick": ["brick wall with mortar lines"],
            "concrete": ["concrete retaining wall"],
            "metal": ["metal fence"],
            "glass": ["glass barrier"]
        }
    }
}

# Lazy singletons
_yolo, _sam, _sam_pred = None, None, None
_clip_model, _clip_preprocess = None, None
_depth_processor, _depth_model = None, None

print("🚀 Initializing AI backends... (models will lazy-load on first use)")

# =========================
# Utilities
# =========================
def _ensure_yolo():
    global _yolo
    if _yolo is None:
        _yolo = YOLO(YOLO_MODEL_PATH)
    return _yolo

def _ensure_sam():
    global _sam, _sam_pred
    if _sam_pred is None:
        _sam = sam_model_registry["vit_t"](checkpoint=SAM_MODEL_PATH).to(DEVICE)
        _sam_pred = SamPredictor(_sam)
    return _sam_pred

def _ensure_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=CLIP_DEVICE)
    return _clip_model, _clip_preprocess

def _ensure_depth():
    global _depth_processor, _depth_model
    if _depth_model is None:
        print("🔧 Initializing depth estimation model (DepthPro)...")
        print(f"🎯 Target device: {DEVICE}")
        
        try:
            print("📥 Loading depth processor...")
            _depth_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
            # _depth_processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
            print(f"✅ Depth processor loaded successfully")
            print(f"📊 Processor config: {_depth_processor.__class__.__name__}")
            
        except Exception as e:
            print(f"❌ Error loading depth processor: {e}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
            raise
        
        try:
            print("🧠 Loading depth model...")
            _depth_model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(DEVICE)
            # _depth_model = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf")
            print(f"✅ Depth model loaded successfully")
            print(f"📊 Model type: {_depth_model.__class__.__name__}")
            
            # Log model parameters before moving to device
            total_params = sum(p.numel() for p in _depth_model.parameters())
            trainable_params = sum(p.numel() for p in _depth_model.parameters() if p.requires_grad)
            print(f"📊 Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
            
            print(f"🚀 Moving model to device: {DEVICE}")
            _depth_model = _depth_model.to(DEVICE)
            print(f"✅ Model moved to device successfully")
            
            print("🔒 Setting model to evaluation mode...")
            _depth_model.eval()
            print(f"✅ Model set to evaluation mode")
            
            # Verify model device placement
            model_device = next(_depth_model.parameters()).device
            print(f"📱 Model device verification: {model_device}")
            
            if torch.cuda.is_available() and DEVICE == "cuda":
                print(f"🔥 GPU memory after model loading:")
                print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                print(f"   Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
                print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"❌ Error loading depth model: {e}")
            import traceback
            print(f"📋 Traceback: {traceback.format_exc()}")
            raise
        
        print("✅ Depth estimation pipeline initialized successfully")
    else:
        print("♻️ Depth model already initialized, reusing existing instance")
        
    return _depth_processor, _depth_model

def preload_all_models():
    """
    Preload all AI models during server startup to avoid lazy loading delays during analysis.
    This will download and initialize YOLO, SAM, CLIP, and DepthPro models.
    """
    print("🚀 Preloading all AI models...")
    
    try:
        print("📦 Loading YOLO model...")
        _ensure_yolo()
        print("✅ YOLO model loaded successfully")
        
        print("📦 Loading SAM model...")
        _ensure_sam()
        print("✅ SAM model loaded successfully")
        
        print("📦 Loading CLIP model...")
        _ensure_clip()
        print("✅ CLIP model loaded successfully")
        
        print("📦 Loading DepthPro model...")
        _ensure_depth()
        print("✅ DepthPro model loaded successfully")
        
        print("🎉 All AI models preloaded successfully!")
        
    except Exception as e:
        print(f"❌ Error during model preloading: {e}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        raise

def _read_image_rgb(path: str) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def _mask_to_base64_png(mask_bool: np.ndarray) -> str:
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    pil = Image.fromarray(mask_u8, mode="L")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _base64_png_to_mask(b64: str) -> np.ndarray:
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))).convert("L"))
    return (arr > 128)

# =========================
# ❗ Material Classification Function
# =========================
def _classify_material(pil_img: Image.Image, cls_id: int):
    clip_model, clip_preprocess = _ensure_clip()

    if cls_id not in CLASS_MATERIAL_PROMPTS:
        return "unknown", 0.0

    prompts_config = CLASS_MATERIAL_PROMPTS[cls_id]
    materials = prompts_config["materials"]

    text_prompts = [tpl for mat_prompts in prompts_config["templates"].values() for tpl in mat_prompts]
    if not text_prompts: return "unknown", 0.0

    text_tokens = clip.tokenize(text_prompts).to(CLIP_DEVICE)
    image_input = clip_preprocess(pil_img).unsqueeze(0).to(CLIP_DEVICE)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    # Map probabilities back to materials
    material_scores = {mat: 0.0 for mat in materials}
    prompt_idx = 0
    for mat in materials:
        num_prompts_for_mat = len(prompts_config["templates"].get(mat, []))
        if num_prompts_for_mat > 0:
            max_prob_for_mat = np.max(probs[prompt_idx : prompt_idx + num_prompts_for_mat])
            material_scores[mat] = max_prob_for_mat
        prompt_idx += num_prompts_for_mat

    if not material_scores: return "unknown", 0.0

    best_material = max(material_scores, key=material_scores.get)
    best_score = material_scores[best_material]

    return best_material, float(best_score)

# =========================
# Part 1: Analysis Pipeline (Hybrid Logic)
# =========================
def run_analysis_pipeline(image_path: str):
    print(f"🧠 Analyzing image with hybrid method: {image_path}")
    yolo, sam_pred = _ensure_yolo(), _ensure_sam()
    image_rgb = _read_image_rgb(image_path)
    H, W = image_rgb.shape[:2]

    print("🔍 YOLO inference...")
    results = yolo(image_path)

    bboxes, class_ids = [], []
    for r in results:
        if getattr(r, "boxes", None) is not None:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                bboxes.append(box.detach().cpu().numpy())
                class_ids.append(int(cls.detach().cpu().numpy()))

    if not bboxes:
        return {"message": "No objects detected", "masks": {}, "material_breakdown": {}}

    print("✂️ SAM segmentation...")
    sam_pred.set_image(image_rgb)
    instance_masks = [sam_pred.predict(box=np.array(box), multimask_output=False)[0].squeeze().astype(bool) for box in bboxes]

    final_class_masks = {}
    material_pixel_counts = {}

    print("🧪 Analyzing materials and calculating percentages...")
    for instance_mask, cls_id in zip(instance_masks, class_ids):
        class_name = CLASS_NAMES.get(cls_id)
        if not class_name: continue

        # 1. Aggregate the main class mask
        if class_name not in final_class_masks:
            final_class_masks[class_name] = np.zeros((H, W), dtype=bool)
        final_class_masks[class_name] |= instance_mask

        # 2. Perform hierarchical segmentation and material analysis
        if cls_id in CLASS_MATERIAL_PROMPTS:
            ys, xs = np.where(instance_mask)
            if ys.size == 0: continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

            cropped_img_rgb = image_rgb[y1:y2, x1:x2]
            cropped_mask = instance_mask[y1:y2, x1:x2]

            sam_pred.set_image(cropped_img_rgb)

            h, w = cropped_mask.shape
            grid_points = [[w * j, h * i] for i in [0.25, 0.5, 0.75] for j in [0.25, 0.5, 0.75] if cropped_mask[int(h*i), int(w*j)]]

            if not grid_points: continue

            if class_name not in material_pixel_counts:
                material_pixel_counts[class_name] = Counter()

            for pt in grid_points:
                part_mask, _, _ = sam_pred.predict(point_coords=np.array([pt]), point_labels=np.array([1]), multimask_output=False)
                part_mask = part_mask.squeeze()

                p_ys, p_xs = np.where(part_mask)
                if p_ys.size == 0: continue
                px1, py1, px2, py2 = p_xs.min(), p_ys.min(), p_xs.max(), p_ys.max()

                material_crop_pil = Image.fromarray(cropped_img_rgb[py1:py2, px1:px2])
                material, score = _classify_material(material_crop_pil, cls_id)

                if score > 0.1: # Confidence threshold
                    pixel_count = np.sum(part_mask)
                    material_pixel_counts[class_name][material] += pixel_count

    # --- Final Output Preparation ---
    print("\n🧾 Encoding masks and calculating percentages...")
    encoded_masks = {cls_name: {"mask": _mask_to_base64_png(mask)} for cls_name, mask in final_class_masks.items()}

    material_breakdown = {}
    for class_name, counts in material_pixel_counts.items():
        total_pixels = sum(counts.values())
        if total_pixels == 0: continue

        breakdown = []
        for material, count in counts.items():
            percentage = (count / total_pixels) * 100
            breakdown.append({"material": material, "percentage": round(percentage, 2)})

        # Sort by percentage
        material_breakdown[class_name] = sorted(breakdown, key=lambda x: x['percentage'], reverse=True)

    result = {
        "detected_classes": [{"id": cid, "name": CLASS_NAMES.get(cid)} for cid in sorted(set(class_ids))],
        "masks": encoded_masks,
        "material_breakdown": material_breakdown,
    }
    print("✅ Hybrid analysis complete.")
    return result

# =========================
# Part 2: Area Calculation
# =========================
def _create_point_cloud(depth_map, fx, fy, cx, cy):
    h, w = depth_map.shape
    v, u = np.mgrid[0:h, 0:w]
    valid = ~np.isnan(depth_map)
    d = depth_map[valid]
    if d.size == 0: return np.empty((0, 3), dtype=np.float32)
    u_valid, v_valid = u[valid], v[valid]
    x = (u_valid - cx) * d / fx
    y = (v_valid - cy) * d / fy
    z = d
    return np.vstack((x, y, z)).T

def run_area_calculation(image_path: str, mask_base64: str, real_distance: float) -> float:
    print(f"📐 Calculating area for image: {image_path}")
    print(f"📊 Input parameters - real_distance: {real_distance}, mask_base64 length: {len(mask_base64) if mask_base64 else 'None'}")
    
    # Load and validate image
    try:
        rgb = _read_image_rgb(image_path)
        H, W = rgb.shape[:2]
        print(f"🖼️ Image loaded successfully - dimensions: {H}x{W}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        raise
    
    # Process mask
    try:
        mask_bool = _base64_png_to_mask(mask_base64)
        print(f"🎭 Original mask shape: {mask_bool.shape}")
        
        if mask_bool.shape != (H, W):
            print(f"🔄 Resizing mask from {mask_bool.shape} to ({H}, {W})")
            mask_bool = cv2.resize(mask_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        mask_pixels = np.sum(mask_bool)
        total_pixels = H * W
        mask_coverage = (mask_pixels / total_pixels) * 100
        print(f"🎯 Mask coverage: {mask_pixels}/{total_pixels} pixels ({mask_coverage:.2f}%)")
        
        if mask_pixels == 0:
            print("⚠️ Warning: Mask is empty (no True pixels)")
            return 0.0
            
    except Exception as e:
        print(f"❌ Error processing mask: {e}")
        raise
    
    # Initialize depth model
    try:
        print("🧠 Loading depth estimation model...")
        depth_processor, depth_model = _ensure_depth()
        print(f"📱 Depth model device: {next(depth_model.parameters()).device}")
    except Exception as e:
        print(f"❌ Error loading depth model: {e}")
        raise
    
    # Generate depth map
    try:
        print("🔍 Running depth estimation...")
        pil_img = Image.fromarray(rgb)
        
        with torch.no_grad():
            inputs = depth_processor(images=pil_img, return_tensors="pt").to(DEVICE)
            print(f"📥 Depth input tensor shape: {inputs['pixel_values'].shape}")
            
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
            print(f"📤 Raw depth output shape: {predicted_depth.shape}")
            
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False,
            ).squeeze()
            print(f"🔄 Interpolated depth shape: {prediction.shape}")
            
    except Exception as e:
        print(f"❌ Error during depth estimation: {e}")
        raise
    
    # Process depth data
    try:
        depth_raw = prediction.cpu().numpy()
        print(f"📊 Raw depth stats - min: {np.nanmin(depth_raw):.4f}, max: {np.nanmax(depth_raw):.4f}, mean: {np.nanmean(depth_raw):.4f}")
        
        depth_masked = np.where(mask_bool, depth_raw, np.nan)
        valid_depth_pixels = np.sum(~np.isnan(depth_masked))
        print(f"🎯 Valid depth pixels in mask: {valid_depth_pixels}/{mask_pixels}")
        
        positive_vals = depth_masked[depth_masked > 0]
        print(f"📈 Positive depth values: {positive_vals.size}/{valid_depth_pixels}")
        
        if positive_vals.size == 0:
            print("⚠️ Warning: No positive depth values found in masked region")
            return 0.0
            
        print(f"📊 Positive depth stats - min: {np.min(positive_vals):.4f}, max: {np.max(positive_vals):.4f}, mean: {np.mean(positive_vals):.4f}")
        
    except Exception as e:
        print(f"❌ Error processing depth data: {e}")
        raise
    
    # Calculate scaling
    try:
        robust_min = np.nanpercentile(positive_vals, 1)
        print(f"📐 Robust minimum depth (1st percentile): {robust_min:.4f}")
        
        if robust_min <= 0 or not np.isfinite(robust_min):
            print(f"❌ Invalid robust minimum: {robust_min}")
            return 0.0
            
        scale = float(real_distance) / float(robust_min) if real_distance else 1.0
        print(f"⚖️ Scaling factor: {scale:.4f} (real_distance: {real_distance}, robust_min: {robust_min})")
        
        depth_scaled = depth_masked * scale
        scaled_positive = depth_scaled[depth_scaled > 0]
        if scaled_positive.size > 0:
            print(f"📏 Scaled depth stats - min: {np.min(scaled_positive):.4f}, max: {np.max(scaled_positive):.4f}, mean: {np.mean(scaled_positive):.4f}")
        
    except Exception as e:
        print(f"❌ Error calculating scaling: {e}")
        raise
    
    # Create point cloud
    try:
        print("☁️ Creating point cloud...")
        focal = float(W)
        cx, cy = W / 2.0, H / 2.0
        print(f"📷 Camera parameters - focal: {focal}, center: ({cx}, {cy})")
        
        pts = _create_point_cloud(depth_scaled, focal, focal, cx, cy)
        print(f"🔢 Point cloud created with {pts.shape[0]} points")
        
        if pts.shape[0] < 3:
            print(f"⚠️ Warning: Insufficient points for mesh creation: {pts.shape[0]} < 3")
            return 0.0
            
        # Log point cloud statistics
        if pts.shape[0] > 0:
            print(f"📊 Point cloud bounds:")
            print(f"   X: [{np.min(pts[:, 0]):.4f}, {np.max(pts[:, 0]):.4f}]")
            print(f"   Y: [{np.min(pts[:, 1]):.4f}, {np.max(pts[:, 1]):.4f}]")
            print(f"   Z: [{np.min(pts[:, 2]):.4f}, {np.max(pts[:, 2]):.4f}]")
            
    except Exception as e:
        print(f"❌ Error creating point cloud: {e}")
        raise
    
    # Create mesh and calculate area
    try:
        print("🕸️ Creating mesh from point cloud...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        print("📐 Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        print(f"📊 Point cloud has {len(pcd.normals)} normals")
        
        print("🔨 Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        
        print(f"🕸️ Mesh created with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        print(f"📏 Density values - min: {np.min(densities):.4f}, max: {np.max(densities):.4f}")
        
        area = float(mesh.get_surface_area())
        print(f"✅ Estimated Surface Area: {area:.2f} m²")
        return area
        
    except Exception as e:
        print(f"❌ Mesh reconstruction error: {e}")
        import traceback
        print(f"📋 Full traceback: {traceback.format_exc()}")
        return 0.0
