from __future__ import annotations

import argparse
import math
import multiprocessing as mp_process
import platform
import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    KEY_COOLDOWN_SEC,
    MARUGOTO_DISTANCE_M,
    MODEL_PATH,
    POSE_CONNECTIONS,
    POSE_KEYBINDS,
    POSE_LABEL_OFFSET_X,
    POSE_LABEL_OFFSET_Y,
    POSE_LABEL_SCALE,
    POSE_STABLE_FRAMES,
)


def create_detector(model_path: Path) -> vision.PoseLandmarker:
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path.resolve()}")

    base_options = python.BaseOptions(model_asset_path=str(model_path.resolve()))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


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


def judge_pose(image, pose_landmarks, pose_world_landmarks=None) -> str | None:
    h, w, _ = image.shape

    nose = pose_landmarks[0]
    left_shoulder = pose_landmarks[11]
    right_shoulder = pose_landmarks[12]
    left_elbow = pose_landmarks[13]
    right_elbow = pose_landmarks[14]
    left_wrist = pose_landmarks[15]
    right_wrist = pose_landmarks[16]

    wrist_distance_m = None
    marugoto = False
    if pose_world_landmarks is not None:
        left_wrist_world = pose_world_landmarks[15]
        right_wrist_world = pose_world_landmarks[16]
        wrist_distance_m = math.sqrt(
            (left_wrist_world.x - right_wrist_world.x) ** 2
            + (left_wrist_world.y - right_wrist_world.y) ** 2
            + (left_wrist_world.z - right_wrist_world.z) ** 2
        )
        marugoto = wrist_distance_m <= MARUGOTO_DISTANCE_M

    left_arm_above_head = (
        left_wrist.y < nose.y and left_elbow.y < left_shoulder.y
    )
    right_arm_above_head = (
        right_wrist.y < nose.y and right_elbow.y < right_shoulder.y
    )
    left_arm_across_face = (
        abs(left_wrist.y - nose.y) < ARM_ACROSS_FACE_Y_THRESHOLD
        and abs(left_elbow.y - nose.y) < ELBOW_ACROSS_FACE_Y_THRESHOLD
        and left_wrist.x > left_shoulder.x
    )
    right_arm_across_face = (
        abs(right_wrist.y - nose.y) < ARM_ACROSS_FACE_Y_THRESHOLD
        and abs(right_elbow.y - nose.y) < ELBOW_ACROSS_FACE_Y_THRESHOLD
        and right_wrist.x < right_shoulder.x
    )
    left_bent = abs(left_wrist.x - left_elbow.x) > ARM_BENT_X_THRESHOLD
    right_bent = abs(right_wrist.x - right_elbow.x) > ARM_BENT_X_THRESHOLD

    image_pose = (
        left_arm_above_head and right_arm_across_face and left_bent and right_bent
    ) or (
        right_arm_above_head and left_arm_across_face and left_bent and right_bent
    )

    if marugoto:
        pose_key = "MARUGOTO"
        label = "MARUGOTO"
        color = (0, 255, 0)
    elif image_pose:
        pose_key = "IMAGE_POSE"
        label = "IMAGE POSE!"
        color = (255, 255, 255)
    elif left_arm_above_head:
        pose_key = "LEFT_ARM_ABOVE_HEAD"
        label = "LEFT ARM ABOVE HEAD"
        color = (255, 255, 128)
    elif right_arm_above_head:
        pose_key = "RIGHT_ARM_ABOVE_HEAD"
        label = "RIGHT ARM ABOVE HEAD"
        color = (255, 255, 0)
    elif left_arm_across_face:
        pose_key = "LEFT_ARM_ACROSS_FACE"
        label = "LEFT ARM ACROSS FACE"
        color = (255, 128, 255)
    elif right_arm_across_face:
        pose_key = "RIGHT_ARM_ACROSS_FACE"
        label = "RIGHT ARM ACROSS FACE"
        color = (255, 128, 128)
    elif left_bent:
        pose_key = "LEFT_BENT"
        label = "LEFT BENT"
        color = (255, 128, 0)
    elif right_bent:
        pose_key = "RIGHT_BENT"
        label = "RIGHT BENT"
        color = (255, 0, 255)
    else:
        pose_key = None
        label = "Keep Trying..."
        color = (0, 0, 255)

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


def send_keypress(key: str) -> None:
    if platform.system() != "Darwin":
        return

    script = f'tell application "System Events" to keystroke "{key}"'
    if key == "space":
        script = 'tell application "System Events" to key code 49'
    elif key == "return":
        script = 'tell application "System Events" to key code 36'

    subprocess.run(["osascript", "-e", script], check=False)


def update_pose_actions(action_state: dict, current_pose: str | None) -> str | None:
    if current_pose is not None and current_pose == action_state["last_pose"]:
        action_state["stable_frames"] += 1
    elif current_pose is not None:
        action_state["last_pose"] = current_pose
        action_state["stable_frames"] = 1
    else:
        action_state["last_pose"] = None
        action_state["stable_frames"] = 0
        return None

    if action_state["stable_frames"] < POSE_STABLE_FRAMES:
        return None

    now = time.time()
    last_trigger = action_state["last_trigger_times"].get(current_pose, 0.0)
    if now - last_trigger < action_state["cooldown_sec"]:
        return None

    key = POSE_KEYBINDS.get(current_pose)
    if key is None:
        return None

    send_keypress(key)
    action_state["last_trigger_times"][current_pose] = now
    action_state["stable_frames"] = 0
    action_state["last_pose"] = None
    action_state["message"] = f"KEY SENT: {current_pose} -> {key}"
    return key


def process_frame(
    image,
    detector: vision.PoseLandmarker,
    combo_state: dict | None,
    action_state: dict | None,
):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = detector.detect(mp_image)
    current_pose = None

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        draw_bones(image, landmarks)
        current_pose = judge_pose(image, landmarks, world_landmarks)

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
        update_pose_actions(action_state, current_pose)
        cv2.putText(
            image,
            action_state["message"],
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            ACTION_LABEL_SCALE,
            (255, 255, 0),
            2,
        )

    return image, current_pose


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
    proc = mp_process.Process(
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

    if queue.empty():
        return False

    return bool(queue.get())


def open_camera(index: int = 0):
    for backend_name in CAMERA_BACKENDS:
        if not probe_camera_backend(index, backend_name, CAMERA_OPEN_TIMEOUT_SEC):
            continue

        if backend_name == "avfoundation" and hasattr(cv2, "CAP_AVFOUNDATION"):
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            opened = cap.isOpened()
            display_name = "AVFoundation"
        else:
            cap = cv2.VideoCapture(index)
            opened = cap.isOpened()
            display_name = "Default"

        if opened:
            return cap, display_name
        cap.release()

    return None, None


def run_image(detector: vision.PoseLandmarker, image_path: Path, output_path: Path | None) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"画像を読めません: {image_path.resolve()}")

    processed, pose = process_frame(image, detector, combo_state=None, action_state=None)
    print(f"detected_pose={pose}")

    if output_path is not None:
        cv2.imwrite(str(output_path), processed)
        print(f"saved={output_path.resolve()}")
        return

    cv2.imshow("MediaPipe Pose", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(detector: vision.PoseLandmarker, camera_index: int) -> None:
    combo_state = {
        "index": 0,
        "last_pose": None,
        "stable_frames": 0,
        "last_step_time": 0.0,
        "message": f"COMBO: 1/{len(COMBO_SEQUENCE)}  {COMBO_SEQUENCE[0]}",
    }
    action_state = {
        "last_pose": None,
        "stable_frames": 0,
        "last_trigger_times": {},
        "cooldown_sec": KEY_COOLDOWN_SEC,
        "message": "KEY READY",
    }

    cap, backend_name = open_camera(camera_index)
    if cap is None:
        raise RuntimeError("カメラを開けませんでした。")

    print(f"camera={backend_name}")
    print("q で終了します")
    print(f"keybinds={POSE_KEYBINDS}")
    if platform.system() == "Darwin":
        print("macOS の場合は Terminal / Python にアクセシビリティ権限が必要です")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("フレーム取得に失敗しました")
                break

            processed, _ = process_frame(
                frame,
                detector,
                combo_state=combo_state,
                action_state=action_state,
            )
            cv2.imshow("MediaPipe Pose", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["camera", "image"], default="camera")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    detector = create_detector(args.model)

    if args.mode == "image":
        run_image(detector, args.image, args.output)
    else:
        run_camera(detector, args.camera_index)


if __name__ == "__main__":
    main()
