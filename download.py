# download.py
import os
import clip

os.makedirs("weights", exist_ok=True)
# This will download ViT-B/32 to ./weights/ViT-B-32.pt if not already present
clip.load("ViT-B/32", device="cpu", download_root="weights")
print("Downloaded CLIP weights to ./weights")
