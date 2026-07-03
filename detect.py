# detect.py — object detection with YOLO + confidence threshold

from ultralytics import YOLO
import os

# a detection is kept only if the model is at least this sure (0.0 to 1.0)
CONFIDENCE_THRESHOLD = 0.5

# 1) load the pre-trained model
model = YOLO("yolov8n.pt")

# 2) find the image in this folder
image_files = [f for f in os.listdir(".") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
filename = image_files[0]
print("Running detection on:", filename)

# 3) run detection, keeping only boxes above our threshold
results = model(filename, conf=CONFIDENCE_THRESHOLD)

# 4) print each detected object
print("\n--- Detections ---")
counts = {}                                    # to count how many of each object
for box in results[0].boxes:
    label = model.names[int(box.cls[0])]       # object name
    confidence = float(box.conf[0])            # how sure (0 to 1)
    print(f"Found: {label}  ({confidence:.0%})")
    counts[label] = counts.get(label, 0) + 1   # add one to this object's count

# 5) print a summary
print("\n--- Summary ---")
if counts:
    for label, n in counts.items():
        print(f"{label}: {n}")
else:
    print("No objects above the confidence threshold.")

# 6) save the annotated image
results[0].save(filename="detected.jpg")
print("\nSaved result as detected.jpg")