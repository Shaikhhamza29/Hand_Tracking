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
cap = cv2.VideoCapture(0)

PINCH_THRESHOLD = 50
PICKUP_RADIUS = 60

# ---------------- FLOWERS (RANDOM) ----------------
NUM_FLOWERS = 7
flowers = []

for _ in range(NUM_FLOWERS):
    flowers.append({
        "x": random.randint(150, 500),
        "y": random.randint(120, 350),
        "z": random.uniform(-0.1, 0.1),

        # Random colors (BGR)
        "petal_color": (
            random.randint(120, 255),
            random.randint(60, 200),
            random.randint(120, 255)
        ),
        "center_color": (
            random.randint(0, 80),
            random.randint(180, 255),
            random.randint(180, 255)
        ),

        # 🌿 Natural tilt: left / right / straight
        "tilt": random.choice([0, -25, 25])
    })

# Only ONE flower can be grabbed
grabbed_index = None

# ---------------- Draw Flower (TILTED STEM) ----------------
def draw_flower(img, center, z, flower, base_size=40):
    x, y = center

    # Depth scaling
    scale = int(base_size - z * 200)
    scale = max(20, min(scale, 90))

    tilt = flower["tilt"]
    rad = math.radians(tilt)

    petal_color = flower["petal_color"]
    center_color = flower["center_color"]

    # ---------------- STEM (TILTED) ----------------
    stem_len = int(scale * 2.5)

    stem_end_x = int(x + math.sin(rad) * stem_len)
    stem_end_y = int(y + math.cos(rad) * stem_len)

    cv2.line(
        img,
        (x, y),
        (stem_end_x, stem_end_y),
        (0, 180, 0),
        4
    )

    # ---------------- LEAVES (ON STEM) ----------------
    leaf_offset = scale // 2

    leaf_x = int(x + math.sin(rad) * (stem_len // 2))
    leaf_y = int(y + math.cos(rad) * (stem_len // 2))

    cv2.ellipse(
        img,
        (leaf_x - leaf_offset, leaf_y),
        (scale // 2, scale // 4),
        tilt - 30,
        0, 360,
        (0, 160, 0),
        -1
    )

    cv2.ellipse(
        img,
        (leaf_x + leaf_offset, leaf_y),
        (scale // 2, scale // 4),
        tilt + 30,
        0, 360,
        (0, 160, 0),
        -1
    )

    # ---------------- FLOWER CENTER ----------------
    cv2.circle(
        img,
        (x, y),
        scale // 4,
        center_color,
        -1
    )

    # ---------------- PETALS (TILTED) ----------------
    petal_radius = scale // 2
    petal_size = scale // 3

    for angle in range(0, 360, 45):
        total_angle = angle + tilt
        rad_p = math.radians(total_angle)

        px = int(x + math.cos(rad_p) * petal_radius)
        py = int(y + math.sin(rad_p) * petal_radius)

        cv2.ellipse(
            img,
            (px, py),
            (petal_size, petal_size // 2),
            total_angle,
            0, 360,
            petal_color,
            -1
        )

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)
            index_z = lm[8].z

            pinch_dist = math.hypot(ix - tx, iy - ty)

            # -------- PICK ONE FLOWER --------
            if pinch_dist < PINCH_THRESHOLD and grabbed_index is None:
                for i, flower in enumerate(flowers):
                    dist = math.hypot(ix - flower["x"], iy - flower["y"])
                    if dist < PICKUP_RADIUS:
                        grabbed_index = i
                        break

            # -------- MOVE SELECTED FLOWER --------
            if grabbed_index is not None:
                flowers[grabbed_index]["x"] = ix
                flowers[grabbed_index]["y"] = iy
                flowers[grabbed_index]["z"] = index_z

            # -------- DROP --------
            if pinch_dist > PINCH_THRESHOLD:
                grabbed_index = None

            # Draw fingers
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 8, (0, 255, 0), -1)
            cv2.line(frame, (tx, ty), (ix, iy), (255, 0, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ---------------- Draw ALL Flowers ----------------
    for flower in flowers:
        draw_flower(frame, (flower["x"], flower["y"]), flower["z"], flower)

    cv2.imshow("AR Flower Bouquet Builder 🌸", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
