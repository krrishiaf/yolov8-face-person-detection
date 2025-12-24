from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import cv2
import os
import time

app = Flask(__name__)

# Folder to store uploads and outputs
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    output_file = None
    is_video = False

    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename

        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # ================= IMAGE HANDLING =================
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            results = model(input_path, save=False)
            img = cv2.imread(input_path)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    f"{label} {conf:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            output_name = f"output_image_{int(time.time())}.jpg"
            output_path = os.path.join(UPLOAD_FOLDER, output_name)
            cv2.imwrite(output_path, img)

            output_file = url_for("static", filename=f"uploads/{output_name}")
            is_video = False

        # ================= VIDEO HANDLING =================
        elif filename.lower().endswith((".mp4", ".avi")):
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                return "Error: Could not open video file"

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if fps is None or fps <= 1:
                fps = 20  # Safe default FPS

            output_name = f"output_video_{int(time.time())}.mp4"
            output_path = os.path.join(UPLOAD_FOLDER, output_name)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)

            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                results = model(frame, save=False)

                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                out.write(frame)

            cap.release()
            out.release()

            if frame_count == 0:
                return "Error: No frames processed"

            output_file = url_for("static", filename=f"uploads/{output_name}")
            is_video = True

    return render_template(
        "index.html",
        output_file=output_file,
        is_video=is_video
    )

if __name__ == "__main__":
    app.run(debug=False)
