import cv2
import mediapipe as mp
import math
import random

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(1)

PINCH_THRESHOLD = 45
PICK_RADIUS = 80

# ---------------- DROP ZONE (CLAW AREA) ----------------
DROP_ZONE = {
    "x1": 420,
    "y1": 260,
    "x2": 640,
    "y2": 480
}

DROP_CENTER_X = (DROP_ZONE["x1"] + DROP_ZONE["x2"]) // 2
DROP_CENTER_Y = (DROP_ZONE["y1"] + DROP_ZONE["y2"]) // 2

def inside_drop_zone(x, y):
    return (
        DROP_ZONE["x1"] <= x <= DROP_ZONE["x2"] and
        DROP_ZONE["y1"] <= y <= DROP_ZONE["y2"]
    )

# ---------------- Load Flower Image ----------------
flower_img = cv2.imread("Images/flower.png", cv2.IMREAD_UNCHANGED)

# ---------------- Alpha Overlay ----------------
def overlay_png(bg, png, x, y, scale):
    png = cv2.resize(png, None, fx=scale, fy=scale)

    h, w = png.shape[:2]
    x1, y1 = int(x - w // 2), int(y - h // 2)
    x2, y2 = x1 + w, y1 + h

    if x2 <= 0 or y2 <= 0 or x1 >= bg.shape[1] or y1 >= bg.shape[0]:
        return bg

    x1_c, y1_c = max(0, x1), max(0, y1)
    x2_c, y2_c = min(bg.shape[1], x2), min(bg.shape[0], y2)

    png = png[y1_c - y1:y2_c - y1, x1_c - x1:x2_c - x1]

    alpha = png[:, :, 3] / 255.0
    for c in range(3):
        bg[y1_c:y2_c, x1_c:x2_c, c] = (
            alpha * png[:, :, c] +
            (1 - alpha) * bg[y1_c:y2_c, x1_c:x2_c, c]
        )
    return bg

# ---------------- Create Flowers ----------------
NUM_FLOWERS = 6
flowers = []

for _ in range(NUM_FLOWERS):
    flowers.append({
        "x": random.randint(150, 350),
        "y": random.randint(200, 350),
        "z": 0.5,
        "base_scale": random.uniform(0.45, 0.6),
        "locked": False,
        "falling": False,
        "target_y": None
    })

grabbed_index = None
was_pinching = False
FALL_SPEED = 12   # pixels per frame

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    pinching = False

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)

        pinch_dist = math.hypot(ix - tx, iy - ty)
        pinching = pinch_dist < PINCH_THRESHOLD

        # -------- PICK (pinch start) --------
        if pinching and not was_pinching:
            for i, f in enumerate(flowers):
                if f["locked"] or f["falling"]:
                    continue
                if math.hypot(ix - f["x"], iy - f["y"]) < PICK_RADIUS:
                    grabbed_index = i
                    break

        # -------- MOVE (while pinching) --------
        if pinching and grabbed_index is not None:
            flowers[grabbed_index]["x"] = ix
            flowers[grabbed_index]["y"] = iy

            z_raw = -lm[8].z
            z_mapped = min(max(z_raw * 2.5, 0.0), 1.0)
            flowers[grabbed_index]["z"] = (
                flowers[grabbed_index]["z"] * 0.7 + z_mapped * 0.3
            )

        # -------- DROP → FALL FROM TOP --------
        if not pinching and was_pinching and grabbed_index is not None:
            f = flowers[grabbed_index]

            if inside_drop_zone(f["x"], f["y"]):
                f["x"] = DROP_CENTER_X
                f["y"] = DROP_ZONE["y1"]   # start at TOP
                f["target_y"] = DROP_CENTER_Y
                f["falling"] = True
                f["locked"] = True

            grabbed_index = None

        mp_draw.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS
        )

    was_pinching = pinching

    # -------- FALL ANIMATION --------
    for f in flowers:
        if f["falling"]:
            f["y"] += FALL_SPEED
            if f["y"] >= f["target_y"]:
                f["y"] = f["target_y"]
                f["falling"] = False

    # -------- DRAW DROP ZONE --------
    cv2.rectangle(
        frame,
        (DROP_ZONE["x1"], DROP_ZONE["y1"]),
        (DROP_ZONE["x2"], DROP_ZONE["y2"]),
        (0, 255, 0),
        2
    )
    cv2.putText(
        frame,
        "DROP AREA",
        (DROP_ZONE["x1"] + 10, DROP_ZONE["y1"] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # -------- DRAW FLOWERS --------
    for f in sorted(flowers, key=lambda x: x["z"]):
        scale = f["base_scale"] + f["z"] * 0.6
        scale = max(0.3, min(scale, 1.2))

        frame = overlay_png(
            frame,
            flower_img,
            f["x"],
            f["y"],
            scale
        )

    cv2.imshow("AR Flower Claw Machine 🌸🕹️", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
