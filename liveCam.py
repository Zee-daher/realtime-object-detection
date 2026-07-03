# liveCam.py — real-time object detection from your webcam

from ultralytics import YOLO
import cv2

# load the pre-trained model
model = YOLO("yolov8n.pt")

# open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Webcam running — press Q to quit.")

while True:
    # read one frame from the camera
    ok, frame = cap.read()
    if not ok:
        print("Could not read from webcam.")
        break

    # run detection on this frame (keep boxes above 50% confidence)
    results = model(frame, conf=0.5, verbose=False)

    # draw the boxes on the frame
    annotated = results[0].plot()

    # show the result in a window
    cv2.imshow("YOLO Live Detection — press Q to quit", annotated)

    # stop when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# clean up
cap.release()
cv2.destroyAllWindows()