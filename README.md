# 🖼️ Object Detection, Image Segmentation, and Material Classification for Urban Surface Analysis

### A Deep Learning Pipeline for Identifying and Analyzing Urban Surfaces Contributing to UHI Effects

This project presents a comprehensive image processing pipeline that integrates object detection (YOLOv8), image segmentation (SAM), and material classification to analyze urban environments. The system identifies key urban components, segments them into precise regions, classifies their surface materials, and calculates their surface areas. The goal is to support Urban Heat Island (UHI) research and smart city analytics through high-resolution, material-level insights.

---

## 🔍 Research Focus

### 📌 Research Question

How can deep learning-based object detection and segmentation be utilized to identify and analyze urban surface materials contributing to Urban Heat Island (UHI) effects?

### 🎯 Objectives

- Detect urban structures and materials in images using object detection (YOLOv8).
- Generate pixel-level segmentation masks for each detected object using SAM.
- Classify segmented regions into surface types.
- Calculate surface area coverage of each material type using mask and bounding box data.

---

## 🧠 Technology Stack

| Component               | Tools/Frameworks                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------ |
| Object Detection        | [YOLOv8](https://github.com/ultralytics/ultralytics)                                 |
| Image Segmentation      | [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) |
| Material Classification | CNN-based or pretrained classifiers                                                  |
| Image Processing        | OpenCV, NumPy                                                                        |
| Area Calculation        | Custom scripts using pixel-to-area scaling                                           |

---
