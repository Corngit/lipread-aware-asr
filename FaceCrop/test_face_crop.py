import cv2
import numpy as np
import subprocess
import os
import glob
import shutil
import tempfile
from ultralytics import YOLO

# =========================
# 參數設定
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-face-lindevs.pt")
TRACKER = "bytetrack.yaml"

TARGET_SIZE = 160
SCALE = 1.5
ALPHA = 0.8

OUTPUT_FPS = 25
OUTPUT_AUDIO_SR = 16000

# =========================
# 載入模型（只載一次）
# =========================
model = YOLO(MODEL_PATH)

# =========================
# 工具函式
# =========================
def run_ffmpeg(cmd):
    subprocess.run(cmd, check=True)

def preprocess_video(input_video, output_video):
    """
    先把原始影片標準化成:
    - 25 fps
    - 16000 Hz
    - mono
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,

        # 視訊標準化成 25fps
        "-vf", f"fps={OUTPUT_FPS}",
        "-r", str(OUTPUT_FPS),

        # 視訊編碼
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",

        # 音訊標準化
        "-c:a", "aac",
        "-ar", str(OUTPUT_AUDIO_SR),
        "-ac", "1",

        output_video
    ]
    run_ffmpeg(cmd)

def build_final_video_from_frames(frame_dir, audio_source_video, output_video):
    """
    用 PNG 幀序列組成最終影片，再接回預處理後影片的音訊
    """
    frame_pattern = os.path.join(frame_dir, "%06d.png")

    cmd = [
        "ffmpeg",
        "-y",

        # 幀序列輸入
        "-framerate", str(OUTPUT_FPS),
        "-i", frame_pattern,

        # 音訊從預處理後影片拿
        "-i", audio_source_video,

        # 視訊輸出規格
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(OUTPUT_FPS),

        # 保險指定尺寸
        "-vf", f"scale={TARGET_SIZE}:{TARGET_SIZE}",

        # 音訊輸出規格
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:a", "aac",
        "-ar", str(OUTPUT_AUDIO_SR),
        "-ac", "1",

        "-shortest",

        output_video
    ]
    run_ffmpeg(cmd)

def crop_square_with_padding(frame, cx, cy, size):
    h, w = frame.shape[:2]
    half = int(size / 2)

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        frame = cv2.copyMakeBorder(
            frame,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REPLICATE
        )
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = frame[y1:y2, x1:x2]

    crop = cv2.resize(
        crop,
        (TARGET_SIZE, TARGET_SIZE),
        interpolation=cv2.INTER_LANCZOS4
    )
    return crop

def process_one_video(input_video, index, total):
    base_name = os.path.splitext(os.path.basename(input_video))[0]

    preprocessed_video = f"__temp_preprocessed_{base_name}.mp4"
    temp_final_video = f"__temp_final_{base_name}.mp4"
    backup_video = f"__backup_{base_name}.mp4"

    print(f"\n[{index}/{total}] 開始處理：{input_video}")

    if not os.path.exists(input_video):
        print(f"找不到檔案，跳過：{input_video}")
        return

    temp_frame_dir = tempfile.mkdtemp(prefix=f"face_crop_{base_name}_")

    try:
        # -------------------------
        # Step 1: 預處理成 25fps + 16kHz + mono
        # -------------------------
        print("  Step 1/3: ffmpeg 預處理...")
        preprocess_video(input_video, preprocessed_video)

        # -------------------------
        # Step 2: YOLO 動態裁切
        # -------------------------
        print("  Step 2/3: YOLO 動態裁切...")
        cap = cv2.VideoCapture(preprocessed_video)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟預處理後影片：{preprocessed_video}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  預處理後 FPS = {fps}")

        smooth_cx = None
        smooth_cy = None
        smooth_size = None
        target_id = None
        last_face_crop = None
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.track(frame, persist=True, tracker=TRACKER, verbose=False)
                boxes = results[0].boxes if results and len(results) > 0 else None

                selected = None

                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()

                    if boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        ids = np.array([-1] * len(xyxy))

                    if target_id is not None and target_id in ids:
                        idx = np.where(ids == target_id)[0][0]
                        selected = xyxy[idx]
                    else:
                        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                        idx = np.argmax(areas)
                        selected = xyxy[idx]

                        if ids[idx] != -1:
                            target_id = ids[idx]

                if selected is not None:
                    x1, y1, x2, y2 = selected
                    w = x2 - x1
                    h = y2 - y1
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0

                    crop_size = max(w, h) * SCALE

                    if smooth_cx is None:
                        smooth_cx, smooth_cy, smooth_size = cx, cy, crop_size
                    else:
                        smooth_cx = ALPHA * smooth_cx + (1 - ALPHA) * cx
                        smooth_cy = ALPHA * smooth_cy + (1 - ALPHA) * cy
                        smooth_size = ALPHA * smooth_size + (1 - ALPHA) * crop_size

                    face_crop = crop_square_with_padding(frame, smooth_cx, smooth_cy, smooth_size)
                    last_face_crop = face_crop.copy()

                else:
                    if last_face_crop is not None:
                        face_crop = last_face_crop
                    else:
                        face_crop = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

                frame_path = os.path.join(temp_frame_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(frame_path, face_crop)
                frame_idx += 1

        finally:
            cap.release()

        print(f"  共輸出 {frame_idx} 張裁切影格")

        # -------------------------
        # Step 3: 組回最終影片
        # -------------------------
        print("  Step 3/3: ffmpeg 合成最終影片...")
        build_final_video_from_frames(temp_frame_dir, preprocessed_video, temp_final_video)

        # -------------------------
        # 覆蓋原檔
        # -------------------------
        os.replace(input_video, backup_video)
        os.replace(temp_final_video, input_video)
        os.remove(backup_video)

        print(f"  完成：{input_video}")

    except FileNotFoundError:
        print("  錯誤：找不到 ffmpeg，請先確認已安裝並加入 PATH。")

    except subprocess.CalledProcessError as e:
        print(f"  ffmpeg 執行失敗：{input_video}")
        print(f"  {e}")

    except Exception as e:
        print(f"  處理失敗：{input_video}")
        print(f"  {e}")

    finally:
        # 清除暫存 PNG 幀
        if os.path.exists(temp_frame_dir):
            try:
                shutil.rmtree(temp_frame_dir)
            except Exception:
                print(f"  警告：無法刪除暫存資料夾 {temp_frame_dir}")

        # 清除暫存影片
        for temp_file in [preprocessed_video, temp_final_video]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    print(f"  警告：無法刪除暫存檔 {temp_file}")

# =========================
# 主程式：抓五位數字檔名 mp4
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

video_pattern = os.path.join(BASE_DIR, "[0-9][0-9][0-9][0-9][0-9].mp4")
video_list = sorted(glob.glob(video_pattern))

if not video_list:
    print("找不到符合規則的影片，例如 00001.mp4")
else:
    print("找到以下影片：")
    for v in video_list:
        print(" ", v)

    total = len(video_list)
    print(f"\n總共 {total} 支影片，開始批次處理...")

    for i, video_file in enumerate(video_list, start=1):
        process_one_video(video_file, i, total)

    print("\n全部處理完成")