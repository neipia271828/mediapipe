from __future__ import annotations

import argparse
import math
import multiprocessing as mp_process
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pynput.keyboard import Key
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController

from constant import (
    ACTION_LABEL_SCALE,
    ARM_ACROSS_FACE_Y_THRESHOLD,
    ARM_BENT_X_THRESHOLD,
    COMBO_LABEL_SCALE,
    COMBO_SEQUENCE,
    COMBO_TIMEOUT_SEC,
    CAMERA_BACKENDS,
    CAMERA_OPEN_TIMEOUT_SEC,
    DEFAULT_IMAGE_PATH,
    DISTANCE_LABEL_SCALE,
    ELBOW_ACROSS_FACE_Y_THRESHOLD,
    LEAN_DEADZONE,
    LEAN_MOUSE_SPEED,
    MARUGOTO_DISTANCE_M,
    MODEL_PATH,
    POSE_CONNECTIONS,
    POSE_LABEL_OFFSET_X,
    POSE_LABEL_OFFSET_Y,
    POSE_LABEL_SCALE,
    POSE_STABLE_FRAMES,
    RUN_GRACE_FRAMES,
    RUN_WRIST_ABOVE_SHOULDER_MARGIN,
    RUN_WRIST_BELOW_SHOULDER_MARGIN,
    SQUAT_BODY_RATIO_THRESHOLD,
    SWING_ARM_TIMEOUT_SEC,
    SWING_COOLDOWN_SEC,
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def create_detector(model_path: Path) -> vision.PoseLandmarker:
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path.resolve()}")

    base_options = python.BaseOptions(model_asset_path=str(model_path.resolve()))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_bones(image, pose_landmarks) -> None:
    h, w, _ = image.shape

    for start_idx, end_idx in POSE_CONNECTIONS:
        start = pose_landmarks[start_idx]
        end = pose_landmarks[end_idx]
        sx, sy = int(start.x * w), int(start.y * h)
        ex, ey = int(end.x * w), int(end.y * h)
        cv2.line(image, (sx, sy), (ex, ey), (255, 255, 255), 2)

    for lm in pose_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)


# ---------------------------------------------------------------------------
# Pose detection
# ---------------------------------------------------------------------------

def judge_pose(image, pose_landmarks, pose_world_landmarks=None) -> str | None:
    h, w, _ = image.shape

    nose           = pose_landmarks[0]
    left_shoulder  = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    left_elbow     = pose_landmarks[13]
    right_elbow    = pose_landmarks[14]
    left_wrist     = pose_landmarks[15]
    right_wrist    = pose_landmarks[16]
    left_hip       = pose_landmarks[23]
    right_hip      = pose_landmarks[24]

    # --- SQUAT: 体の縦方向が肩幅に対して圧縮されている ---
    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_mid_y      = (left_hip.y + right_hip.y) / 2
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)
    body_ratio = (hip_mid_y - shoulder_mid_y) / max(shoulder_width, 0.01)
    squat = body_ratio < SQUAT_BODY_RATIO_THRESHOLD

    # --- RUN: 片腕が肩より上、反対腕が肩より下 ---
    right_arm_up   = right_wrist.y < right_shoulder.y - RUN_WRIST_ABOVE_SHOULDER_MARGIN
    left_arm_down  = left_wrist.y  > left_shoulder.y  + RUN_WRIST_BELOW_SHOULDER_MARGIN
    left_arm_up    = left_wrist.y  < left_shoulder.y  - RUN_WRIST_ABOVE_SHOULDER_MARGIN
    right_arm_down = right_wrist.y > right_shoulder.y + RUN_WRIST_BELOW_SHOULDER_MARGIN
    run = (right_arm_up and left_arm_down) or (left_arm_up and right_arm_down)

    # --- 既存ポーズ ---
    wrist_distance_m = None
    marugoto = False
    if pose_world_landmarks is not None:
        left_wrist_world  = pose_world_landmarks[15]
        right_wrist_world = pose_world_landmarks[16]
        wrist_distance_m = math.sqrt(
            (left_wrist_world.x - right_wrist_world.x) ** 2
            + (left_wrist_world.y - right_wrist_world.y) ** 2
            + (left_wrist_world.z - right_wrist_world.z) ** 2
        )
        marugoto = wrist_distance_m <= MARUGOTO_DISTANCE_M

    left_arm_above_head  = left_wrist.y  < nose.y and left_elbow.y  < left_shoulder.y
    right_arm_above_head = right_wrist.y < nose.y and right_elbow.y < right_shoulder.y
    left_arm_across_face = (
        abs(left_wrist.y  - nose.y) < ARM_ACROSS_FACE_Y_THRESHOLD
        and abs(left_elbow.y  - nose.y) < ELBOW_ACROSS_FACE_Y_THRESHOLD
        and left_wrist.x > left_shoulder.x
    )
    right_arm_across_face = (
        abs(right_wrist.y - nose.y) < ARM_ACROSS_FACE_Y_THRESHOLD
        and abs(right_elbow.y - nose.y) < ELBOW_ACROSS_FACE_Y_THRESHOLD
        and right_wrist.x < right_shoulder.x
    )
    left_bent  = abs(left_wrist.x  - left_elbow.x)  > ARM_BENT_X_THRESHOLD
    right_bent = abs(right_wrist.x - right_elbow.x) > ARM_BENT_X_THRESHOLD

    image_pose = (
        left_arm_above_head and right_arm_across_face and left_bent and right_bent
    ) or (
        right_arm_above_head and left_arm_across_face and left_bent and right_bent
    )

    # --- 優先順位でポーズを決定 ---
    if squat:
        pose_key, label, color = "SQUAT",           "SQUAT",             (0, 128, 255)
    elif run:
        pose_key, label, color = "RUN",             "RUN",               (0, 255, 128)
    elif marugoto:
        pose_key, label, color = "MARUGOTO",        "MARUGOTO",          (0, 255, 0)
    elif image_pose:
        pose_key, label, color = "IMAGE_POSE",      "IMAGE POSE!",       (255, 255, 255)
    elif left_arm_above_head:
        pose_key, label, color = "LEFT_ARM_ABOVE_HEAD",  "LEFT ARM ABOVE HEAD",  (255, 255, 128)
    elif right_arm_above_head:
        pose_key, label, color = "RIGHT_ARM_ABOVE_HEAD", "RIGHT ARM ABOVE HEAD", (255, 255, 0)
    elif left_arm_across_face:
        pose_key, label, color = "LEFT_ARM_ACROSS_FACE",  "LEFT ARM ACROSS FACE",  (255, 128, 255)
    elif right_arm_across_face:
        pose_key, label, color = "RIGHT_ARM_ACROSS_FACE", "RIGHT ARM ACROSS FACE", (255, 128, 128)
    elif left_bent:
        pose_key, label, color = "LEFT_BENT",  "LEFT BENT",  (255, 128, 0)
    elif right_bent:
        pose_key, label, color = "RIGHT_BENT", "RIGHT BENT", (255, 0, 255)
    else:
        pose_key, label, color = None, "Keep Trying...", (0, 0, 255)

    chest_x = int((left_shoulder.x + right_shoulder.x) / 2 * w) - POSE_LABEL_OFFSET_X
    chest_y = int(min(left_shoulder.y, right_shoulder.y) * h) - POSE_LABEL_OFFSET_Y
    cv2.putText(
        image,
        label,
        (max(chest_x, 10), max(chest_y, 40)),
        cv2.FONT_HERSHEY_SIMPLEX,
        POSE_LABEL_SCALE,
        color,
        3,
    )

    if wrist_distance_m is not None:
        cv2.putText(
            image,
            f"PALM DIST: {wrist_distance_m:.2f}m",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            DISTANCE_LABEL_SCALE,
            (255, 255, 255),
            2,
        )

    return pose_key


def get_body_lean(pose_landmarks) -> float:
    """
    体の傾きを返す。
    正値 = 画像内で右側に傾いている（カメラ非ミラー時はユーザーの右）
    負値 = 左側に傾いている
    """
    left_shoulder  = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    left_hip       = pose_landmarks[23]
    right_hip      = pose_landmarks[24]

    shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_mid_x      = (left_hip.x      + right_hip.x)      / 2

    # 肩の中心が腰の中心より右 = 右に傾いている
    return shoulder_mid_x - hip_mid_x


# ---------------------------------------------------------------------------
# Combo (既存機能)
# ---------------------------------------------------------------------------

def update_combo(combo_state: dict, current_pose: str | None) -> None:
    if current_pose is not None and current_pose == combo_state["last_pose"]:
        combo_state["stable_frames"] += 1
    elif current_pose is not None:
        combo_state["last_pose"] = current_pose
        combo_state["stable_frames"] = 1
    else:
        combo_state["last_pose"] = None
        combo_state["stable_frames"] = 0

    now = time.time()
    if combo_state["index"] > 0 and now - combo_state["last_step_time"] > COMBO_TIMEOUT_SEC:
        combo_state["index"] = 0
        combo_state["message"] = f"COMBO RESET: 1/{len(COMBO_SEQUENCE)}  {COMBO_SEQUENCE[0]}"

    if current_pose is None or combo_state["stable_frames"] < POSE_STABLE_FRAMES:
        return

    expected_pose = COMBO_SEQUENCE[combo_state["index"]]
    if current_pose == expected_pose:
        combo_state["index"] += 1
        combo_state["last_step_time"] = now
        combo_state["last_pose"] = None
        combo_state["stable_frames"] = 0

        if combo_state["index"] == len(COMBO_SEQUENCE):
            combo_state["message"] = "COMBO SUCCESS!"
            combo_state["index"] = 0
        else:
            next_pose = COMBO_SEQUENCE[combo_state["index"]]
            combo_state["message"] = (
                f"COMBO: {combo_state['index'] + 1}/{len(COMBO_SEQUENCE)}  {next_pose}"
            )
    elif current_pose == COMBO_SEQUENCE[0]:
        combo_state["index"] = 1
        combo_state["last_step_time"] = now
        combo_state["last_pose"] = None
        combo_state["stable_frames"] = 0
        next_pose = COMBO_SEQUENCE[combo_state["index"]]
        combo_state["message"] = (
            f"COMBO: {combo_state['index'] + 1}/{len(COMBO_SEQUENCE)}  {next_pose}"
        )
    else:
        combo_state["index"] = 0
        combo_state["last_pose"] = None
        combo_state["stable_frames"] = 0
        combo_state["message"] = f"WRONG ORDER: 1/{len(COMBO_SEQUENCE)}  {COMBO_SEQUENCE[0]}"


# ---------------------------------------------------------------------------
# Input control: 長押し + マウス移動
# ---------------------------------------------------------------------------

def _set_held(held_keys: set, key, should_hold: bool, kb: KeyboardController) -> None:
    """キーの長押し / 解放を管理する。

    pynput の kb.press() は1回のキーダウンイベントしか送らないため、
    毎フレーム送り続けてゲーム側に「押しっぱなし」を認識させる。
    """
    if should_hold:
        kb.press(key)          # 毎フレーム送信 → ゲームが長押しとして認識
        held_keys.add(key)
    elif key in held_keys:
        kb.release(key)
        held_keys.discard(key)


def update_held_inputs(
    action_state: dict,
    current_pose: str | None,
    lean: float,
    kb: KeyboardController,
    mouse_ctrl: MouseController,
) -> None:
    """
    ポーズに応じてキー長押し / 解放と、傾きに応じたマウス移動を行う。

    RUN / SQUAT がグレース期間内に再検出されればキー押下が継続する。
    これにより腕振りの折り返し時にキーが一瞬解放されるのを防ぐ。
    """
    # stable_frames カウント
    if current_pose is not None and current_pose == action_state["last_pose"]:
        action_state["stable_frames"] += 1
    elif current_pose is not None:
        action_state["last_pose"] = current_pose
        action_state["stable_frames"] = 1
    else:
        action_state["last_pose"] = None
        action_state["stable_frames"] = 0

    stable = action_state["stable_frames"] >= POSE_STABLE_FRAMES
    held   = action_state["held_keys"]

    # --- RUN グレース管理 ---
    if stable and current_pose == "RUN":
        action_state["run_grace"] = RUN_GRACE_FRAMES  # ポーズ検出中はフル充填
    elif action_state["run_grace"] > 0:
        action_state["run_grace"] -= 1                # 途切れたらカウントダウン

    # --- SQUAT グレース管理 ---
    if stable and current_pose == "SQUAT":
        action_state["squat_grace"] = RUN_GRACE_FRAMES
    elif action_state["squat_grace"] > 0:
        action_state["squat_grace"] -= 1

    should_run   = action_state["run_grace"]   > 0
    should_squat = action_state["squat_grace"] > 0

    _set_held(held, "d",       should_run,   kb)
    _set_held(held, Key.shift, should_squat, kb)

    # マウス移動: RUN 中に傾いたらカーソルを動かす
    if should_run:
        if lean > LEAN_DEADZONE:
            dx = int((lean - LEAN_DEADZONE) * LEAN_MOUSE_SPEED)
            mouse_ctrl.move(dx, 0)
            action_state["message"] = f"RUN + LEAN RIGHT  dx={dx}px (lean={lean:+.3f})"
        elif lean < -LEAN_DEADZONE:
            dx = int((lean + LEAN_DEADZONE) * LEAN_MOUSE_SPEED)
            mouse_ctrl.move(dx, 0)
            action_state["message"] = f"RUN + LEAN LEFT   dx={dx}px (lean={lean:+.3f})"
        else:
            action_state["message"] = "RUN"
    elif should_squat:
        action_state["message"] = "SQUAT [Shift]"
    elif current_pose is not None:
        action_state["message"] = f"POSE: {current_pose}"
    else:
        action_state["message"] = "READY"


def release_all_held(action_state: dict, kb: KeyboardController) -> None:
    """プログラム終了時にすべての長押しキーを解放する。"""
    for key in list(action_state.get("held_keys", set())):
        try:
            kb.release(key)
        except Exception:
            pass
    action_state.get("held_keys", set()).clear()


# ---------------------------------------------------------------------------
# Motion gesture detection
# ---------------------------------------------------------------------------

def update_motion_gestures(
    gesture_state: dict,
    right_wrist_x: float,
    right_wrist_y: float,
    midline_x: float,
    shoulder_y: float,
    mouse_ctrl: MouseController,
) -> str | None:
    """
    ゾーン遷移でジェスチャーを検出する。

    SWING_USE   : (正中線より右 + 肩より上) → (正中線より左 + 肩より下) → 左クリック
    SWING_SCROLL: (正中線より左 + 肩より下) → (正中線より右 + 肩より下) → ホイール
    """
    now = time.time()

    right_of_mid = right_wrist_x > midline_x
    left_of_mid  = right_wrist_x < midline_x
    above_sh     = right_wrist_y < shoulder_y
    below_sh     = right_wrist_y > shoulder_y

    # ---- SWING_USE ステートマシン ----
    if gesture_state["use_state"] == "IDLE":
        # 開始ゾーン: 正中線より右 + 肩より上
        if right_of_mid and above_sh:
            gesture_state["use_state"]    = "ARMED"
            gesture_state["use_arm_time"] = now
    else:  # ARMED
        if now - gesture_state["use_arm_time"] > SWING_ARM_TIMEOUT_SEC:
            gesture_state["use_state"] = "IDLE"   # タイムアウト
        elif left_of_mid and below_sh:
            # 終了ゾーン到達 → 左クリック
            last = gesture_state["last_trigger_times"].get("SWING_USE", 0.0)
            gesture_state["use_state"] = "IDLE"
            if now - last >= SWING_COOLDOWN_SEC:
                gesture_state["last_trigger_times"]["SWING_USE"] = now
                gesture_state["scroll_state"] = "IDLE"  # SCROLL の誤発火防止
                mouse_ctrl.click(Button.left)
                gesture_state["message"] = "USE [Left Click]"
                return "SWING_USE"

    # ---- SWING_SCROLL ステートマシン ----
    if gesture_state["scroll_state"] == "IDLE":
        # 開始ゾーン: 正中線より左 + 肩より下
        if left_of_mid and below_sh:
            gesture_state["scroll_state"]    = "ARMED"
            gesture_state["scroll_arm_time"] = now
    else:  # ARMED
        if now - gesture_state["scroll_arm_time"] > SWING_ARM_TIMEOUT_SEC:
            gesture_state["scroll_state"] = "IDLE"
        elif right_of_mid and below_sh:
            # 終了ゾーン到達 → ホイール
            last = gesture_state["last_trigger_times"].get("SWING_SCROLL", 0.0)
            gesture_state["scroll_state"] = "IDLE"
            if now - last >= SWING_COOLDOWN_SEC:
                gesture_state["last_trigger_times"]["SWING_SCROLL"] = now
                mouse_ctrl.scroll(0, -3)
                gesture_state["message"] = "SCROLL [Wheel]"
                return "SWING_SCROLL"

    return None


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------

def process_frame(
    image,
    detector: vision.PoseLandmarker,
    combo_state: dict | None,
    action_state: dict | None,
) -> tuple:
    """
    Returns: (processed_image, current_pose, lean, right_wrist_xy, midline_x, shoulder_y)
    lean=0.0, right_wrist_xy=None, midline_x=0.5, shoulder_y=0.5 when no pose detected.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result    = detector.detect(mp_image)

    current_pose   = None
    lean           = 0.0
    right_wrist_xy = None
    midline_x      = 0.5
    shoulder_y     = 0.5

    if result.pose_landmarks:
        landmarks       = result.pose_landmarks[0]
        world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

        draw_bones(image, landmarks)
        current_pose   = judge_pose(image, landmarks, world_landmarks)
        lean           = get_body_lean(landmarks)
        rw             = landmarks[16]
        right_wrist_xy = (rw.x, rw.y)
        ls, rs         = landmarks[11], landmarks[12]
        midline_x      = (ls.x + rs.x) / 2
        shoulder_y     = (ls.y + rs.y) / 2

    if combo_state is not None:
        update_combo(combo_state, current_pose)
        cv2.putText(
            image,
            combo_state["message"],
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            COMBO_LABEL_SCALE,
            (0, 255, 0),
            2,
        )

    if action_state is not None:
        cv2.putText(
            image,
            action_state["message"],
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            ACTION_LABEL_SCALE,
            (255, 255, 0),
            2,
        )

    return image, current_pose, lean, right_wrist_xy, midline_x, shoulder_y


# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------

def _probe_camera_worker(index: int, backend_name: str, queue) -> None:
    if backend_name == "avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)

    opened = cap.isOpened()
    if opened:
        queue.put(True)
    cap.release()


def probe_camera_backend(index: int, backend_name: str, timeout_sec: float) -> bool:
    queue = mp_process.Queue()
    proc  = mp_process.Process(
        target=_probe_camera_worker,
        args=(index, backend_name, queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_sec)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False

    return not queue.empty() and bool(queue.get())


def open_camera(index: int = 0):
    for backend_name in CAMERA_BACKENDS:
        if not probe_camera_backend(index, backend_name, CAMERA_OPEN_TIMEOUT_SEC):
            continue

        if backend_name == "avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap          = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            display_name = "AVFoundation"
        else:
            cap          = cv2.VideoCapture(index)
            display_name = "Default"

        if cap.isOpened():
            return cap, display_name
        cap.release()

    return None, None


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_image(detector: vision.PoseLandmarker, image_path: Path, output_path: Path | None) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像を読めません: {image_path.resolve()}")

    processed, pose, _, _, _, _ = process_frame(image, detector, combo_state=None, action_state=None)
    print(f"detected_pose={pose}")

    if output_path is not None:
        cv2.imwrite(str(output_path), processed)
        print(f"saved={output_path.resolve()}")
        return

    cv2.imshow("MediaPipe Pose", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(detector: vision.PoseLandmarker, camera_index: int) -> None:
    kb         = KeyboardController()
    mouse_ctrl = MouseController()

    combo_state = {
        "index":          0,
        "last_pose":      None,
        "stable_frames":  0,
        "last_step_time": 0.0,
        "message":        f"COMBO: 1/{len(COMBO_SEQUENCE)}  {COMBO_SEQUENCE[0]}",
    }
    action_state = {
        "last_pose":     None,
        "stable_frames": 0,
        "held_keys":     set(),
        "run_grace":     0,
        "squat_grace":   0,
        "message":       "READY",
    }
    gesture_state = {
        "use_state":          "IDLE",   # "IDLE" | "ARMED"
        "use_arm_time":       0.0,
        "scroll_state":       "IDLE",
        "scroll_arm_time":    0.0,
        "last_trigger_times": {},
        "message":            "",
    }

    cap, backend_name = open_camera(camera_index)
    if cap is None:
        raise RuntimeError("カメラを開けませんでした。")

    print(f"camera={backend_name}")
    print("q で終了します")
    print("ポーズ対応:")
    print("  RUN   (片腕↑ + 反対腕↓)     → d キー長押し")
    print("  RUN + 右傾き                → d 長押し + カーソル右")
    print("  RUN + 左傾き                → d 長押し + カーソル左")
    print("  右手を右上→左下に振り下ろす  → 左クリック")
    print("  右手を左下→右下へ横移動      → マウスホイール")
    print("  しゃがむ                    → Shift 長押し")
    print("macOS の場合は Terminal に Accessibility 権限が必要です")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("フレーム取得に失敗しました")
                break

            processed, current_pose, lean, right_wrist_xy, midline_x, shoulder_y = process_frame(
                frame,
                detector,
                combo_state=combo_state,
                action_state=action_state,
            )

            # キー長押し / マウス移動
            update_held_inputs(action_state, current_pose, lean, kb, mouse_ctrl)

            # モーションジェスチャー検出
            if right_wrist_xy is not None:
                update_motion_gestures(
                    gesture_state,
                    right_wrist_xy[0], right_wrist_xy[1],
                    midline_x, shoulder_y,
                    mouse_ctrl,
                )

            # ジェスチャーメッセージをフレームに描画
            gesture_msg = str(gesture_state["message"])
            if gesture_msg:
                fh = processed.shape[0]
                cv2.putText(
                    processed,
                    gesture_msg,
                    (20, fh - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 255),
                    2,
                )

            cv2.imshow("MediaPipe Pose", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        release_all_held(action_state, kb)
        cap.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",         choices=["camera", "image"], default="camera")
    parser.add_argument("--image",        type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output",       type=Path)
    parser.add_argument("--camera-index", type=int,  default=0)
    parser.add_argument("--model",        type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main():
    args     = parse_args()
    detector = create_detector(args.model)

    if args.mode == "image":
        run_image(detector, args.image, args.output)
    else:
        run_camera(detector, args.camera_index)


if __name__ == "__main__":
    main()
