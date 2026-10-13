import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # pretrained on COCO (80 classes, incl. "cell phone"); auto-downloads ~6 MB
cap = cv2.VideoCapture(0)    # 0 = default webcam

while True:
    ok, frame = cap.read()             # frame is a NumPy array: height x width x 3 (BGR)
    if not ok:
        break                          # camera failed or video ended
    result = model(frame, verbose=False)[0]   # run detection; [0] because it returns one result per image
    cv2.imshow("test", result.plot())  # .plot() draws all boxes + labels for us
    if cv2.waitKey(1) & 0xFF == ord("q"):     # needed to refresh the window; press q to quit
        break

cap.release()
cv2.destroyAllWindows()