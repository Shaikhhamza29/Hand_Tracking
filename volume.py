import cv2
import mediapipe as mp
import math
import random

# ================= MediaPipe Setup =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ================= Camera =================
cap = cv2.VideoCapture(1)

# ================= Config =================
PINCH_THRESHOLD = 45
PICK_RADIUS = 80
MIN_DROP_SCALE = 0.75
FALL_SPEED = 12
GRASS_Y_RATIO = 0.65

# ================= Drop Zone =================
DROP_ZONE = {"x1": 450, "y1": 360, "x2": 640, "y2": 480}
DROP_CENTER_X = (DROP_ZONE["x1"] + DROP_ZONE["x2"]) // 2
DROP_CENTER_Y = (DROP_ZONE["y1"] + DROP_ZONE["y2"]) // 2
DROP_MARGIN = 60   # 👈 NEW (near box allowed)

def near_drop_zone(x, y):
    return (
        DROP_ZONE["x1"] - DROP_MARGIN <= x <= DROP_ZONE["x2"] + DROP_MARGIN and
        DROP_ZONE["y1"] - DROP_MARGIN <= y <= DROP_ZONE["y2"] + DROP_MARGIN
    )

# ================= Load Images =================
flower_images = [
    cv2.imread("Images/flower.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("Images/flower2.png", cv2.IMREAD_UNCHANGED),
    cv2.imread("Images/flower3.png", cv2.IMREAD_UNCHANGED),
]
flower_images = [img for img in flower_images if img is not None]
if not flower_images:
    raise RuntimeError("No flower images loaded")

grass_img = cv2.imread("Images/grass.png", cv2.IMREAD_UNCHANGED)
if grass_img is None:
    raise RuntimeError("Grass image not found")

# ================= Helpers =================
def overlay_png(bg, png, x, y, scale):
    if png is None or scale <= 0:
        return bg

    png = cv2.resize(png, None, fx=scale, fy=scale)
    h, w = png.shape[:2]
    if h == 0 or w == 0:
        return bg

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

def draw_grass(frame, grass_img):
    h, w = frame.shape[:2]
    scale = w / grass_img.shape[1]
    grass_h = int(grass_img.shape[0] * scale)
    y_pos = h - grass_h
    overlay_png(frame, grass_img, w // 2, y_pos + grass_h // 2, scale)

def draw_claw_arm(frame, x, y):
    cv2.line(frame, (x, 0), (x, y - 25), (180, 180, 180), 3)
    cv2.circle(frame, (x, y - 15), 8, (120, 120, 120), -1)
    cv2.line(frame, (x - 10, y - 5), (x - 20, y + 10), (160, 160, 160), 2)
    cv2.line(frame, (x + 10, y - 5), (x + 20, y + 10), (160, 160, 160), 2)

def draw_drop_pit(frame):
    # dark pit
    cv2.rectangle(frame,
        (DROP_ZONE["x1"], DROP_ZONE["y1"]),
        (DROP_ZONE["x2"], DROP_ZONE["y2"]),
        (10, 10, 10), -1
    )
    # rim
    cv2.rectangle(frame,
        (DROP_ZONE["x1"], DROP_ZONE["y1"]),
        (DROP_ZONE["x2"], DROP_ZONE["y2"]),
        (0, 200, 0), 4
    )

# ================= Create Flowers =================
NUM_FLOWERS = 6
flowers = []

GROUND_Y = int(480 * GRASS_Y_RATIO) + 60

for _ in range(NUM_FLOWERS):
    # mostly small, few bigger
    if random.random() < 0.7:      # 70% small flowers
        base_scale = random.uniform(0.22, 0.30)
    else:                          # 30% slightly bigger
        base_scale = random.uniform(0.32, 0.42)

    flowers.append({
        "x": random.randint(150, 350),
        "y": GROUND_Y + random.randint(-6, 6),
        "z": 0.15,
        "base_scale": base_scale,
        "img": random.choice(flower_images),
        "drop_scale": None,  # 👈 store scale at release

        "picked": False,
        "locked": False,
        "falling": False,
        "target_y": None,

        "visible": True  # 👈 NEW
    })

grabbed_index = None
was_pinching = False
claw_x, claw_y = None, None
score = 0

# ================= Main Loop =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    draw_grass(frame, grass_img)
    draw_drop_pit(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    pinching = False

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark
        ix, iy = int(lm[8].x * w), int(lm[8].y * h)
        tx, ty = int(lm[4].x * w), int(lm[4].y * h)

        claw_x, claw_y = ix, iy
        pinching = math.hypot(ix - tx, iy - ty) < PINCH_THRESHOLD

        if pinching and not was_pinching:
            for i, f in enumerate(flowers):
                if f["falling"] or f["locked"]:
                    continue
                if math.hypot(ix - f["x"], iy - f["y"]) < PICK_RADIUS:
                    grabbed_index = i
                    f["picked"] = True
                    f["z"] = 0.8
                    break

        if pinching and grabbed_index is not None:
            f = flowers[grabbed_index]
            f["x"], f["y"] = ix, iy
            z_raw = -lm[8].z
            f["z"] = f["z"] * 0.7 + min(max(z_raw * 2.5, 0.0), 1.0) * 0.3

        # ---- DROP & SCORE ----
        if not pinching and was_pinching and grabbed_index is not None:
            f = flowers[grabbed_index]

            # 🔐 lock the size at drop time
            f["drop_scale"] = f["base_scale"] + f["z"] * 0.8

            if near_drop_zone(f["x"], f["y"]):
                f["visible"] = False
                f["locked"] = True
                score += 1
            else:
                f["target_y"] = GROUND_Y
                f["picked"] = False
                f["z"] = 0.2

            f["falling"] = True
            grabbed_index = None

    was_pinching = pinching

    if claw_x and claw_y:
        draw_claw_arm(frame, claw_x, claw_y)
    for f in flowers:
        if f["falling"] and f["target_y"] is not None:
            f["y"] += FALL_SPEED
            if f["y"] >= f["target_y"]:
                f["y"] = f["target_y"]
                f["falling"] = False
                f["target_y"] = None

    for f in sorted(flowers, key=lambda x: x["z"]):
        if not f["visible"]:
            continue

        if f["picked"]:
            scale = f["base_scale"] + f["z"] * 0.8
        elif f["falling"] and f["drop_scale"] is not None:
            scale = f["drop_scale"]  # 👈 stays BIG while falling
        else:
            scale = f["base_scale"]  # small on grass

        frame = overlay_png(frame, f["img"], f["x"], f["y"], scale)

    cv2.putText(frame, f"Score: {score}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AR Flower Claw Machine 🌸🕹️", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
