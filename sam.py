import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from skimage import morphology  # For mask post-processing
import os

# 🔹 Set image path (Change this to use a different image)
IMAGE_PATH = "resource/img5.png"

# Create output directories
os.makedirs("output/detection", exist_ok=True)
os.makedirs("output/segmentation", exist_ok=True)
os.makedirs("output/extracted_objects", exist_ok=True)

# Load YOLO model
model = YOLO('best.pt')

# Run inference
results = model(IMAGE_PATH)

# Get bounding boxes and class labels from YOLO detections
bboxes = []
class_ids = []
for r in results:
    if hasattr(r, 'boxes') and r.boxes is not None:
        for box, class_id in zip(r.boxes.xyxy, r.boxes.cls):
            bboxes.append(box.cpu().numpy())  # Convert to NumPy
            class_ids.append(int(class_id.cpu().numpy()))  # Convert to integer

# Convert bounding boxes to NumPy array with float32 type
if not bboxes:
    raise ValueError("No bounding boxes detected by YOLO.")
input_boxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)

# Save object detection visualization
detection_img = results[0].plot()  # Get image with bounding boxes
detection_filename = f"output/detection/{os.path.basename(IMAGE_PATH)}"
cv2.imwrite(detection_filename, detection_img)

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

# Move SAM model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device)

# Initialize SAM predictor
predictor = SamPredictor(sam)

# Load and preprocess the image
image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Set SAM input image
predictor.set_image(image)

# Prepare input prompts (bounding boxes in this case)
input_prompts = []
for box in input_boxes:
    input_prompts.append({
        "box": box,  # Bounding box in [x1, y1, x2, y2] format
        "point_coords": None,  # No point prompts
        "point_labels": None,  # No point prompts
    })

# Get segmentation masks
masks = []
for prompt in input_prompts:
    mask, _, _ = predictor.predict(
        point_coords=prompt["point_coords"],
        point_labels=prompt["point_labels"],
        box=prompt["box"],  # Pass the bounding box
        multimask_output=False,
    )
    mask = mask.squeeze()  # Remove batch dimension

    # Post-process mask (e.g., remove small regions)
    mask = morphology.remove_small_objects(mask, min_size=100)  # Adjust min_size as needed
    mask = morphology.remove_small_holes(mask, area_threshold=100)  # Adjust area_threshold as needed

    masks.append(mask)

# Define a color map for classes
class_colors = {
    0: [1, 0, 0, 0.5],  # Red for class 0 (e.g., "flats")
    1: [0, 0, 1, 0.5],  # Blue for class 1 (e.g., "vehicles")
    2: [1, 1, 0, 0.5],  # Yellow for class 2 (e.g., "construction")
    3: [0, 1, 0, 0.5],  # Green for class 3 (e.g., "natures")
}

# Choose which class(es) to extract
EXTRACT_CLASSES = [2]  # Change this list to extract specific objects

# Extract & Save Selected Objects
extracted_images = []
for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
    if class_id in EXTRACT_CLASSES:
        # Apply mask to extract the object
        extracted_object = np.zeros_like(image)
        extracted_object[mask > 0] = image[mask > 0]
        
        # Save the extracted object image
        filename = f"output/extracted_objects/class_{class_id}_{i}.png"
        cv2.imwrite(filename, cv2.cvtColor(extracted_object, cv2.COLOR_RGB2BGR))
        extracted_images.append(filename)

# Create segmentation visualization
plt.figure(figsize=(10, 10))
plt.imshow(image)

# Overlay segmentation masks with class-wise colors
for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
    # Get the color for the current class
    color = class_colors.get(class_id, [1, 1, 1, 0.5])  # Default to white if class not found

    # Create a colored mask using the class color
    colored_mask = np.zeros((*mask.shape, 4))  # RGBA format
    colored_mask[mask > 0] = color  # Assign color based on class
    plt.imshow(colored_mask, alpha=0.5)

plt.axis("off")

# Save the segmented output
segmentation_filename = f"output/segmentation/{os.path.basename(IMAGE_PATH)}"
plt.savefig(segmentation_filename, bbox_inches="tight", pad_inches=0, dpi=300)
plt.close()  # Close the figure to free memory

print(f"Results saved in:")
print(f"- Object detection: {detection_filename}")
print(f"- Segmentation: {segmentation_filename}")
print(f"- Extracted {len(extracted_images)} objects in 'output/extracted_objects/'")