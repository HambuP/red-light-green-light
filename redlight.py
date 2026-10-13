import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    finish_x = int(w * 0.75)
    cv2.line(frame, (finish_x, 0), (finish_x, h), (0, 215, 255), 3)

    result = model(frame, classes=[0], conf=0.5, verbose=False)[0]

    if len(result.boxes) == 0:
        cv2.putText(frame, "No player", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        center_x = (x1 + x2) // 2
        cv2.circle(frame, (center_x, (y1 + y2) // 2), 8, (0, 0, 255), -1)
        if center_x > finish_x:
            cv2.putText(frame, "YOU WIN!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Red Light, Green Light", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()