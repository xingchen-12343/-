"""
小动物视频检测流水线
流程: 视频读取 → 背景减除(MOG2) → 形态学闭运算合并碎片 → 轮廓合并 → 整体框标注
"""

import cv2
import numpy as np
import argparse
import sys
import os


def merge_contours(contours, merge_dist=80):
    """将距离较近的轮廓合并为一个整体检测框"""
    if not contours:
        return []

    # 获取所有轮廓的外接矩形
    rects = [cv2.boundingRect(cnt) for cnt in contours]

    # 不断合并距离近的矩形，直到没有可合并的
    merged = True
    while merged:
        merged = False
        new_rects = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            x1, y1, w1, h1 = rects[i]
            cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
            group = [rects[i]]
            used[i] = True
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                x2, y2, w2, h2 = rects[j]
                cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
                # 判断两个矩形中心点距离是否小于阈值
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                if dist < merge_dist:
                    group.append(rects[j])
                    used[j] = True
                    merged = True
            # 将同组矩形合并为一个大外接矩形
            gx = min(r[0] for r in group)
            gy = min(r[1] for r in group)
            gx2 = max(r[0] + r[2] for r in group)
            gy2 = max(r[1] + r[3] for r in group)
            new_rects.append((gx, gy, gx2 - gx, gy2 - gy))
            # 更新中心点为合并后的中心
            cx1, cy1 = gx + (gx2 - gx) / 2, gy + (gy2 - gy) / 2
        rects = new_rects

    return rects


def draw_boxes(frame, rects):
    """在原帧上画框标注整只动物"""
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Animal", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def detect_animals(input_path, output_path, min_area=50, max_area=200000, merge_dist=80):
    """主流水线: 背景减除 + 轮廓合并，识别整只动物"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"输入: {input_path}")
    print(f"输出: {output_path}")
    print(f"分辨率: {width}x{height}, 帧率: {fps:.1f}, 总帧数: {total_frames}")
    print(f"面积范围: [{min_area}, {max_area}], 合并距离: {merge_dist}")

    # 背景减除器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    # 大核闭运算: 将动物碎片合并为一个连通区域
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    frame_count = 0
    detect_count = 0
    max_boxes = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 背景减除
        fg_mask = bg_subtractor.apply(frame)

        # 去除阴影
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # 开运算去噪
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
        # 大核闭运算: 将同一动物的碎片连通起来
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        # 轮廓检测
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 按面积筛选
        filtered = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                filtered.append(cnt)

        # 合并相邻轮廓为一个整体框
        merged_rects = merge_contours(filtered, merge_dist)

        if merged_rects:
            detect_count += 1
            if len(merged_rects) > max_boxes:
                max_boxes = len(merged_rects)
            # 检测到多个框时打印帧号和数量
            if len(merged_rects) > 1:
                print(f"  [!] 第 {frame_count} 帧检测到 {len(merged_rects)} 个框: {merged_rects}")

        result = draw_boxes(frame.copy(), merged_rects)
        out.write(result)

        if frame_count % 100 == 0:
            print(f"  已处理 {frame_count}/{total_frames} 帧...")

    cap.release()
    out.release()
    print(f"完成! 共处理 {frame_count} 帧, 检测到动物 {detect_count} 帧")
    print(f"单帧最多框数: {max_boxes}")
    print(f"输出视频: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="小动物视频检测 - 背景减除+轮廓合并")
    parser.add_argument("--input", "-i", required=True, help="输入视频文件路径")
    parser.add_argument("--output", "-o", default="output.mp4", help="输出视频文件路径 (默认: output.mp4)")
    parser.add_argument("--min-area", type=int, default=50, help="最小轮廓面积 (默认: 50)")
    parser.add_argument("--max-area", type=int, default=200000, help="最大轮廓面积 (默认: 200000)")
    parser.add_argument("--merge-dist", type=int, default=80, help="轮廓合并距离/像素 (默认: 80)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在 {args.input}")
        sys.exit(1)

    detect_animals(args.input, args.output, args.min_area, args.max_area, args.merge_dist)


if __name__ == "__main__":
    main()
