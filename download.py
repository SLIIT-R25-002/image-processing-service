# download.py
import os
import clip
import torch
from transformers import AutoModelForDepthEstimation

os.makedirs("weights", exist_ok=True)

# Download CLIP model
print("Downloading CLIP weights...")
clip.load("ViT-B/32", device="cpu", download_root="weights")
print("✅ Downloaded CLIP weights to ./weights")

# Download Depth-Anything-V2 model
print("Downloading Depth-Anything-V2 model...")
depth_model = AutoModelForDepthEstimation.from_pretrained(
    "depth-anything/Depth-Anything-V2-Base-hf",
    ignore_mismatched_sizes=True
)
# Save the model weights
depth_model_path = "weights/depth_anything_v2_vitb.pth"
torch.save(depth_model.state_dict(), depth_model_path)
print(f"✅ Downloaded Depth-Anything-V2 weights to {depth_model_path}")
