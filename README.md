# YOLOv8 Image & Video Detection Web App

A Flask-based web application that performs human detection on images and videos using YOLOv8.
Supports image preview and downloadable processed videos with bounding boxes.

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

## Demo
Step 1: Download or clone this GitHub repository and open the folder on your system.

Step 2: Open a terminal inside the project folder by going to codespace and type git clone https://github.com/krrishiaf/yolov8-face-person-detection.git .
Step 3: Run pip install -r requirements.txt to install everything needed.
Step 4: Run python app.py to start the project.If you still see some import error no worries just paste this 
sudo apt update
sudo apt install -y libgl1
what this will do? basically it will download leftover requirements for the AI to work.Now just go and run python app.py again
Step 5: Now there it will show like WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.* Running on http://127.0.0.1:5000 .⬅️now click on this link

Step 6: After clicking on the link, you will be redirected to a website, and that's all. Now just sit back comfortably🙂 and click on upload an image to see the detection result on the screen, or upload a video to get a processed video file for download.

Step 7: For large videos, it is recommended to run the project locally, as GitHub Codespaces has a file size limit that cannot exceed 15mb, so I recommend using vs code for that .

