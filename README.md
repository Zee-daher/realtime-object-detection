# Real-Time Object Detection with YOLOv8

A hands-on computer vision project that detects objects in images and in a live webcam feed, built step by step while learning the fundamentals of computer vision.

![demo](demo.gif)

## About

This project uses **YOLOv8n** (the nano version of YOLOv8) to perform object detection — locating objects in an image and drawing a labeled bounding box around each one, together with a confidence score. YOLOv8n was chosen because it is the lightest and fastest model in the family, which makes it well suited for **real-time** detection where speed matters more than maximum accuracy.

The project is organized as a small learning journey, from understanding how a computer "sees" an image all the way to real-time detection.

## The journey (how it was built)

**1 — `show_image.py`: an image is a matrix of numbers**
Opens an image with OpenCV and prints its shape and raw pixel values. This makes a core idea concrete: to a computer, an image is just a grid of numbers (height x width x color channels), and every computer vision operation is math performed on that grid.

**2 — `detect.py`: object detection on a still image**
Loads a pre-trained YOLOv8n model and runs detection on an image. It applies a **confidence threshold** to keep only reliable detections, prints each object found with its confidence, prints a summary count per class, and saves an annotated image with the bounding boxes drawn on it.

**3 — `liveCam.py`: real-time detection from a webcam**
Captures frames from the webcam in a loop and runs detection on each frame, drawing the boxes live. This demonstrates the core structure of any real-time vision system: read a frame, run inference, display the result, and repeat many times per second.

## Concepts applied

- **Transfer learning** — using a model pre-trained on the COCO dataset instead of training from scratch.
- **Object detection vs. classification** — not just *what* is in the image, but *where* (bounding boxes).
- **Confidence threshold** — filtering out weak, unreliable detections.
- **Real-time inference** — processing a live video stream frame by frame.
- **Speed vs. accuracy trade-off** — choosing the nano model for smooth real-time performance.

## Run it locally

\`\`\`bash
# 1. create an environment and install dependencies
conda create -n cv python=3.11 -y
conda activate cv
pip install ultralytics

# 2. detect objects in an image (put an image in the folder first)
python detect.py

# 3. run real-time detection from your webcam (press Q to quit)
python liveCam.py
\`\`\`

The YOLOv8n model file (\`yolov8n.pt\`) downloads automatically the first time you run the code.

## Tech stack

Python · Ultralytics YOLOv8 · OpenCV · PyTorch

## Possible next steps

- [ ] Detect objects in a video file and save the annotated output
- [ ] Train the model on a custom dataset
- [ ] Deploy on an edge device (e.g. NVIDIA Jetson, Raspberry Pi)
