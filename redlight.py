import time
import random
import subprocess
import cv2
from ultralytics import YOLO

# --- Tuning knobs ---
GREEN_MIN, GREEN_MAX = 1.0, 2.0   # how long GREEN lasts (random between these, in seconds)
RED_MIN, RED_MAX = 2.5, 4.5       # how long RED lasts
MOVE_THRESHOLD = 20               # pixels you may move during RED before you're out
GRACE_PERIOD = 1.5                # seconds to hear "red light" and actually stop
START_HOLD = 2.0                  # seconds to stand in the start box before the game begins
START_ZONE_X = 0.25               # start box = left 25% of the screen
FINISH_X = 0.75                   # finish line at 75% of the screen
VOICE_RATE = "250"                # speaking speed for the Mac voice (default is ~175)
WINDOW = "Red Light, Green Light"

click = None   # last mouse click position, filled in by on_mouse

def on_mouse(event, x, y, flags, param):
    global click
    if event == cv2.EVENT_LBUTTONDOWN:
        click = (x, y)

def speak(text):
    subprocess.Popen(["say", "-r", VOICE_RATE, text])   # runs in the background, doesn't block

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cv2.namedWindow(WINDOW)
cv2.setMouseCallback(WINDOW, on_mouse)

state = "WAITING"      # WAITING -> PLAYING -> WON or ELIMINATED
light = "GREEN"
next_switch = 0
red_start = 0
frozen_x = None
in_zone_since = None   # time when the player entered the start box

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()

    finish_x = int(w * FINISH_X)
    zone_x = int(w * START_ZONE_X)
    cv2.line(frame, (finish_x, 0), (finish_x, h), (0, 215, 255), 3)

    # --- Lights (only switch while playing) ---
    if state == "PLAYING" and now >= next_switch:
        if light == "GREEN":
            light = "RED"
            red_start = now
            frozen_x = None
            next_switch = now + random.uniform(RED_MIN, RED_MAX)
        else:
            light = "GREEN"
            next_switch = now + random.uniform(GREEN_MIN, GREEN_MAX)
        speak(light + " light")

    # --- Detection ---
    result = model(frame, classes=[0], conf=0.5, verbose=False)[0]

    if len(result.boxes) == 0:
        in_zone_since = None
        cv2.putText(frame, "No player", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        x1, y1, x2, y2 = result.boxes[0].xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center_x = (x1 + x2) // 2
        cv2.circle(frame, (center_x, (y1 + y2) // 2), 8, (0, 0, 255), -1)

        if state == "WAITING":
            if center_x < zone_x:
                if in_zone_since is None:
                    in_zone_since = now
                elif now - in_zone_since >= START_HOLD:
                    state = "PLAYING"
                    light = "GREEN"
                    next_switch = now + random.uniform(GREEN_MIN, GREEN_MAX)
                    speak("Go! Green light")
            else:
                in_zone_since = None

        elif state == "PLAYING":
            if light == "RED" and now - red_start > GRACE_PERIOD:
                if frozen_x is None:
                    frozen_x = center_x
                elif abs(center_x - frozen_x) > MOVE_THRESHOLD:
                    state = "ELIMINATED"
                    speak("Eliminated")
            if state == "PLAYING" and center_x > finish_x:
                state = "WON"
                speak("You win")

    # --- Screen text for each state ---
    if state == "WAITING":
        cv2.rectangle(frame, (0, 0), (zone_x, h), (255, 255, 0), 3)
        if in_zone_since is None:
            msg = "Walk into the START box"
        else:
            msg = f"Hold still... {START_HOLD - (now - in_zone_since):.1f}"
        cv2.putText(frame, msg, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    elif state == "PLAYING":
        if light == "GREEN":
            light_color = (0, 255, 0)
        else:
            light_color = (0, 0, 255)
        cv2.putText(frame, light + " LIGHT", (w // 2 - 130, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, light_color, 4)
    elif state == "WON":
        cv2.putText(frame, "YOU WIN!", (w // 2 - 160, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    elif state == "ELIMINATED":
        cv2.putText(frame, "ELIMINATED", (w // 2 - 220, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # --- Restart button ---
    bx1, by1, bx2, by2 = w - 210, 10, w - 10, 60
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (60, 60, 60), -1)
    cv2.putText(frame, "RESTART", (bx1 + 30, by1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    restart = False
    if click is not None:
        cx, cy = click
        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
            restart = True
        click = None

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        restart = True

    if restart:
        state = "WAITING"
        light = "GREEN"
        frozen_x = None
        in_zone_since = None

cap.release()
cv2.destroyAllWindows()