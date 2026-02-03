---
title: Pothole Detection
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🚗 Pothole Detection System

An AI-powered system to detect potholes in road images using deep learning.

## Model Details

- **Architecture:** Faster R-CNN with ResNet50-FPN backbone
- **Input Size:** 640x640
- **Classes:** Pothole (binary detection)

## Performance Metrics

| Metric | Score |
|--------|-------|
| AP@50 | 62.9% |
| AP@75 | 52.0% |
| mAP@[0.5:0.95] | 42.0% |
| Precision | 45.5% |
| Recall | 76.5% |
| F1-Score | 57.1% |

## Usage

1. Upload a road image
2. Adjust confidence threshold (default: 0.5)
3. View detected potholes with bounding boxes

## Training

- Dataset: Custom pothole dataset (665 images, 1740 annotations)
- Epochs: 15
- Optimizer: SGD with momentum
- Data Augmentation: Flip, brightness, noise, shadow, etc.
