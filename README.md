# YOLOv8 Image & Video Detection Web App

An AI-powered web application built using **YOLOv8** and **Flask** that performs human detection on both images and videos.

## 🚀 Features
- Upload images or videos
- Detect humans using YOLOv8
- Draw bounding boxes on detected persons
- Download processed videos with detections
- Clean and user-friendly web interface

## 🧠 Tech Stack
- Python
- YOLOv8 (Ultralytics)
- Flask
- OpenCV
- HTML (Jinja2 templates)

## ⚙️ How It Works
1. User uploads an image or video
2. YOLOv8 processes each frame
3. Bounding boxes are drawn on detected humans
4. Output image is displayed directly
5. Output video is provided as a downloadable file (for browser compatibility)

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py
