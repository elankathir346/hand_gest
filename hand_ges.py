"""
Scrolling Using Hand Gestures
- Open palm  -> scroll up
- Closed fist -> scroll down
- Thumb-index distance controls scroll speed
Press 'q' to quit.

Install:  pip install opencv-python mediapipe pyautogui
"""

import time

import cv2
import mediapipe as mp
import pyautogui

# ---------- Settings ----------
MIN_SCROLL_SPEED = 20     # scroll amount when thumb and index are close
MAX_SCROLL_SPEED = 200    # scroll amount when thumb and index are far apart
SCROLL_DELAY = 0.05       # seconds between scroll actions
CAMERA_INDEX = 0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Landmark ids: (fingertip, pip joint) for index, middle, ring, pinky
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]


def count_raised_fingers(lm):
    """Count raised fingers (excluding thumb). A finger is up if its tip
    is higher on screen (smaller y) than its pip joint."""
    return sum(1 for tip, pip in FINGERS if lm[tip].y < lm[pip].y)


def get_gesture(lm):
    fingers = count_raised_fingers(lm)
    if fingers == 4:
        return "Open Palm"
    if fingers == 0:
        return "Closed Fist"
    return "None"


def pinch_ratio(lm):
    """Thumb-index distance divided by hand size (wrist to middle knuckle),
    so it works at any distance from the camera."""
    def dist(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    hand_size = dist(lm[0], lm[9])
    return dist(lm[4], lm[8]) / hand_size if hand_size else 0


def scroll_speed_from_ratio(ratio):
    """Map ratio (about 0.1 to 1.2) to a scroll speed."""
    t = max(0.0, min(1.0, (ratio - 0.1) / 1.1))
    return int(MIN_SCROLL_SPEED + t * (MAX_SCROLL_SPEED - MIN_SCROLL_SPEED))


def bounding_box(lm, w, h, pad=20):
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    return (int(min(xs)) - pad, int(min(ys)) - pad,
            int(max(xs)) + pad, int(max(ys)) + pad)


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    last_scroll = 0.0
    prev_time = time.time()

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture, speed, hand_label = "None", 0, "-"

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark
                hand_label = results.multi_handedness[0].classification[0].label

                gesture = get_gesture(lm)
                speed = scroll_speed_from_ratio(pinch_ratio(lm))

                # Scroll with delay
                now = time.time()
                if gesture != "None" and now - last_scroll >= SCROLL_DELAY:
                    pyautogui.scroll(speed if gesture == "Open Palm" else -speed)
                    last_scroll = now

                # Visual feedback
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                x1, y1, x2, y2 = bounding_box(lm, w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(frame,
                         (int(lm[4].x * w), int(lm[4].y * h)),
                         (int(lm[8].x * w), int(lm[8].y * h)),
                         (255, 0, 255), 2)

            # FPS
            cur_time = time.time()
            fps = 1 / (cur_time - prev_time) if cur_time != prev_time else 0
            prev_time = cur_time

            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Hand: {hand_label}   Speed: {speed}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Scrolling", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()