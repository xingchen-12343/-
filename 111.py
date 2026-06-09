import cv2
import numpy as np
from collections import deque
from pathlib import Path

# ====================== 【重要】这里必须改成你真实的视频文件路径 ======================
# Windows 路径三种正确写法：
# 1. 使用双反斜杠 \\
# 2. 使用原始字符串 r''
# 3. 使用正斜杠 /
VIDEO_PATH = r'D:\数字图像课设\屏幕录制 2026-06-03 120620.mp4'  # 必须是视频文件！不是文件夹
OUT_VIDEO = 'rat_detection_result.mp4'
OUT_DIR = Path('detection_screenshots')
OUT_DIR.mkdir(exist_ok=True)

# 参数（已针对黑夜环境调优）
THRESH = 10
MIN_AREA = 80
MAX_AREA_RATIO = 0.35

# CLAHE 暗光增强参数
CLAHE_CLIP = 3.0
CLAHE_GRID = (8, 8)

# 自适应阈值相关
ADAPTIVE_BLOCK = 31       # 自适应阈值邻域大小（奇数）
ADAPTIVE_OFFSET = 5       # 自适应阈值偏移量
USE_ADAPTIVE = True       # 是否启用自适应阈值（黑夜推荐 True）

# MOG2 背景建模参数
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 36
MOG2_SHADOW_DETECT = False  # 关闭阴影检测，避免误过滤暗区目标

# 帧累积缓冲长度（捕获慢速运动）
ACCUM_LEN = 5

# 预创建形态学核（避免每帧重复分配）
KERNEL_ERODE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
KERNEL_DILATE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
KERNEL_CLOSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 复用 CLAHE 对象
_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)


def preprocess(frame):
    """预处理：转灰度 → 降噪 → CLAHE 暗光增强"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 暗光 sensor 噪声抑制
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # CLAHE 自适应直方图均衡化，提升暗区对比度
    gray = _clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def three_frame_diff(g1, g2, g3):
    """三帧差分 + 自适应阈值"""
    d1 = cv2.absdiff(g2, g1)
    d2 = cv2.absdiff(g3, g2)

    if USE_ADAPTIVE:
        # 自适应阈值：根据局部亮度自动调整，黑夜中更鲁棒
        b1 = cv2.adaptiveThreshold(d1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, ADAPTIVE_BLOCK, -ADAPTIVE_OFFSET)
        b2 = cv2.adaptiveThreshold(d2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, ADAPTIVE_BLOCK, -ADAPTIVE_OFFSET)
    else:
        _, b1 = cv2.threshold(d1, THRESH, 255, cv2.THRESH_BINARY)
        _, b2 = cv2.threshold(d2, THRESH, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_and(b1, b2)

    # 形态学处理：用更大的核保护小目标
    mask = cv2.erode(mask, KERNEL_ERODE_3, iterations=1)
    mask = cv2.dilate(mask, KERNEL_DILATE_5, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE_7, iterations=2)

    return mask


def mog2_detect(mog2, frame):
    """MOG2 背景建模检测运动区域"""
    fg_mask = mog2.apply(frame)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.erode(fg_mask, KERNEL_DILATE_5, iterations=1)
    fg_mask = cv2.dilate(fg_mask, KERNEL_DILATE_5, iterations=3)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, KERNEL_DILATE_5, iterations=2)
    return fg_mask


def accumulate_mask(mask_buffer, new_mask):
    """帧累积：对多帧 mask 做 OR，捕获慢速运动"""
    mask_buffer.append(new_mask)
    accumulated = None
    for m in mask_buffer:
        accumulated = m if accumulated is None else cv2.bitwise_or(accumulated, m)
    return accumulated



def detect_targets(mask, frame):
    h, w = frame.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA_RATIO * h * w:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        ratio = bw / max(bh, 1)

        if 0.25 <= ratio <= 5 and bw >= 8 and bh >= 8:
            boxes.append((x, y, bw, bh, area))

    return boxes


def merge_masks(mask_diff, mask_mog2):
    """融合三帧差分 mask 和 MOG2 mask，取 OR 并做后处理"""
    merged = cv2.bitwise_or(mask_diff, mask_mog2)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, KERNEL_CLOSE_5, iterations=2)
    return merged


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"错误：无法打开视频文件，请检查路径是否正确：{VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, fps, (width, height))

    # MOG2 背景建模器
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_SHADOW_DETECT
    )

    # 初始化三帧
    frames = []
    for _ in range(3):
        ret, frame = cap.read()
        if not ret:
            print("错误：视频帧数不足或读取失败！")
            cap.release()
            writer.release()
            return
        frames.append(frame)

    f1, f2, f3 = frames
    g1, g2, g3 = preprocess(f1), preprocess(f2), preprocess(f3)

    # 让 MOG2 先学习几帧背景
    for f in frames:
        mog2.apply(f)

    # 帧累积缓冲（deque 自动丢弃旧帧）
    mask_buffer = deque(maxlen=ACCUM_LEN + 1)

    frame_idx = 2
    saved = 0
    print_interval = max(int(fps * 5), 30)  # 每 5 秒或每 30 帧输出一次进度

    while True:
        # 双通道检测
        mask_diff = three_frame_diff(g1, g2, g3)
        mask_mog2 = mog2_detect(mog2, f3)

        # 融合两种 mask
        mask = merge_masks(mask_diff, mask_mog2)

        # 帧累积，捕获慢速运动
        mask = accumulate_mask(mask_buffer, mask)

        boxes = detect_targets(mask, f3)

        annotated = f3.copy()

        if boxes:
            for x, y, w, h, area in boxes:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, "mouse",
                            (x, max(20, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

            cv2.putText(annotated, "Intrusion detected",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

            # 每秒保存一张截图
            if saved < 8 and frame_idx % int(max(fps, 1)) == 0:
                cv2.imwrite(str(OUT_DIR / f'detect_{frame_idx}.jpg'), annotated)
                cv2.imwrite(str(OUT_DIR / f'mask_{frame_idx}.jpg'), mask)
                saved += 1

        writer.write(annotated)

        # 进度输出
        if frame_idx % print_interval == 0:
            pct = frame_idx / total_frames * 100 if total_frames else 0
            print(f"\r处理进度: {frame_idx}/{total_frames} 帧 ({pct:.1f}%)", end="", flush=True)

        # 读取下一帧
        ret, next_frame = cap.read()
        if not ret:
            break

        g1, g2, g3 = g2, g3, preprocess(next_frame)
        f3 = next_frame
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"\n处理完成！共 {frame_idx} 帧，输出视频：{OUT_VIDEO}")


if __name__ == "__main__":
    main()