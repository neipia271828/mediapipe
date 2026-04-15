from pathlib import Path


MODEL_PATH = Path("pose_landmarker.task")
DEFAULT_IMAGE_PATH = Path("photos/image.png")

COMBO_SEQUENCE = [
    "LEFT_ARM_ABOVE_HEAD",
    "RIGHT_ARM_ACROSS_FACE",
    "IMAGE_POSE",
]

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

# --- 既存ポーズ閾値 ---
MARUGOTO_DISTANCE_M = 0.20
ARM_ACROSS_FACE_Y_THRESHOLD = 0.18
ELBOW_ACROSS_FACE_Y_THRESHOLD = 0.22
ARM_BENT_X_THRESHOLD = 0.05

# --- RUN ポーズ閾値 ---
# 腕が肩より上: wrist.y < shoulder.y - この値
RUN_WRIST_ABOVE_SHOULDER_MARGIN = 0.05
# 腕が腰より下: wrist.y > hip.y + この値
RUN_WRIST_BELOW_HIP_MARGIN = 0.02

# --- SQUAT 閾値 ---
# (hip_mid_y - shoulder_mid_y) / shoulder_width < この値 = しゃがみ検出
# 直立時は約 0.9〜1.1。しゃがむと体が縮むので値が小さくなる
SQUAT_BODY_RATIO_THRESHOLD = 0.65

# --- 重心傾き / マウス移動 ---
LEAN_DEADZONE = 0.04        # この値未満の傾きは無視（中立）
LEAN_MOUSE_SPEED = 8000      # 傾き量に掛ける倍率 (lean_amount * LEAN_MOUSE_SPEED = px/frame)

# --- モーションジェスチャー閾値 ---
SWING_COOLDOWN_SEC = 0.5
# 開始ゾーンに入ってからこの秒数以内に終了ゾーンへ移動しないとリセット
SWING_ARM_TIMEOUT_SEC = 1.5

# --- 表示 ---
POSE_LABEL_SCALE = 1.0
POSE_LABEL_OFFSET_X = 120
POSE_LABEL_OFFSET_Y = 20
DISTANCE_LABEL_SCALE = 0.7
COMBO_LABEL_SCALE = 0.9
ACTION_LABEL_SCALE = 0.8

# --- タイミング ---
COMBO_TIMEOUT_SEC = 3.0
POSE_STABLE_FRAMES = 3
KEY_COOLDOWN_SEC = 1.0
# RUN/SQUAT ポーズが途切れても d/Shift を保持し続けるフレーム数
# 腕振りの折り返し時に一瞬ポーズが外れても継続する
RUN_GRACE_FRAMES = 24

# --- カメラ ---
CAMERA_BACKENDS = [
    "default",
    "avfoundation",
]
CAMERA_OPEN_TIMEOUT_SEC = 2.0
