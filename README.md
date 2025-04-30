# 🖼️ Object Detection, Image Segmentation, and Material Classification for Urban Surface Analysis

### A Deep Learning Pipeline for Identifying and Analyzing Urban Surfaces Contributing to UHI Effects

This project presents a comprehensive image processing pipeline that integrates object detection (YOLOv8), image segmentation (SAM), and material classification to analyze urban environments. The system identifies key urban components (e.g., buildings, pavements, rooftops), segments them into precise regions, classifies their surface materials (e.g., concrete, asphalt, vegetation), and calculates their surface areas. The goal is to support Urban Heat Island (UHI) research and smart city analytics through high-resolution, material-level insights.

---

## 🔍 Research Focus

### 📌 Research Question

How can deep learning-based object detection and segmentation be utilized to identify and analyze urban surface materials contributing to Urban Heat Island (UHI) effects?

### 🎯 Objectives

- Detect urban structures and materials in images using object detection (YOLOv8).
- Generate pixel-level segmentation masks for each detected object using SAM.
- Classify segmented regions into surface types (e.g., asphalt, vegetation, concrete).
- Calculate surface area coverage of each material type using mask and bounding box data.
- Store processed data in a structured format for integration with IoT-based urban monitoring systems.
- (Planned) Develop a mobile application to streamline image capture and analysis in real-world environments.

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
