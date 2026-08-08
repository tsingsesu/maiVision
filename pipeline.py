# -*- coding: utf-8 -*-
"""
pipeline.py —— 数据准备流水线

把「原始标注数据」变成「YOLO 训练集」的完整流程:
  1. 统一图片尺寸 (缩放+黑边 → 1280x720)
  2. 转换标注坐标 (跟随图片缩放/平移)
  3. 离线数据增强 (30张 → 180张, 含坐标变换)
  4. 启动训练

用法:
  python pipeline.py
   (需先设置下方 SOURCE_IMG / SOURCE_LABEL 为你的原始数据目录)

依赖: cv2, numpy, ultralytics
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# 配置区 (改这里, 不用改下面函数)
# ============================================================
# 原始数据: 从 Label Studio 导出的、已标注好的图片和txt
SOURCE_IMG = r"C:\Users\lhl87\Desktop\yolo_train\images\train"      # 原始图片
SOURCE_LABEL = r"C:\Users\lhl87\Desktop\yolo_train\labels\train"    # 原始标注

# 处理后数据输出到这些目录 (与原始分开, 不覆盖)
RESIZED_IMG = r"C:\Users\lhl87\Desktop\yolo_train\images\train_resized"
RESIZED_LABEL = r"C:\Users\lhl87\Desktop\yolo_train\labels\train_resized"
AUG_IMG = r"C:\Users\lhl87\Desktop\yolo_train\images\train_aug"
AUG_LABEL = r"C:\Users\lhl87\Desktop\yolo_train\labels\train_aug"

# 目标尺寸 (和官方数据集一致 1280x720)
TARGET_W, TARGET_H = 1280, 720

# 数据配置 yaml
DATA_YAML = r"C:\Users\lhl87\Desktop\maiVision\maidata.yaml"


def read_boxes(txt_path):
    """读取 YOLO 标注: 每行 [cls, cx, cy, w, h] 返回列表"""
    boxes = []
    for line in open(txt_path):
        p = line.strip().split()
        if len(p) == 5:
            boxes.append([float(x) for x in p])
    return boxes


def write_boxes(txt_path, boxes):
    """把框列表写回 YOLO 格式 txt"""
    with open(txt_path, "w") as f:
        for b in boxes:
            f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")


def resize_and_pad(img_path, out_path):
    """步骤1: 图片缩放到目标尺寸, 保持宽高比, 黑边填充居中

    返回: (scale, ox, oy) 缩放比例和偏移, 供坐标转换用
    """
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 长边缩放比例: 让图不超过目标画布
    scale = min(TARGET_W / w, TARGET_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 创建目标尺寸画布(黑边), 图放中间
    ox = (TARGET_W - new_w) // 2
    oy = (TARGET_H - new_h) // 2
    canvas = np.full((TARGET_H, TARGET_W, 3), 0, dtype=np.uint8)
    canvas[oy:oy + new_h, ox:ox + new_w] = resized

    cv2.imwrite(out_path, canvas)
    return scale, ox, oy


def convert_box(box, orig_w, orig_h, scale, ox, oy):
    """把单个框从'原图坐标'转到'缩放后图坐标'

    原归一化坐标 -> 原图像素 -> 缩放+平移 -> 新图像素 -> 新图归一化
    """
    cls, cx, cy, w, h = box
    # 归一化 -> 原图像素
    x1, y1 = (cx - w / 2) * orig_w, (cy - h / 2) * orig_h
    x2, y2 = (cx + w / 2) * orig_w, (cy + h / 2) * orig_h
    # 缩放 + 平移 -> 新图像素
    x1n, y1n = x1 * scale + ox, y1 * scale + oy
    x2n, y2n = x2 * scale + ox, y2 * scale + oy
    # 新图像素 -> 新图归一化
    cx_n = (x1n + x2n) / 2 / TARGET_W
    cy_n = (y1n + y2n) / 2 / TARGET_H
    w_n = (x2n - x1n) / TARGET_W
    h_n = (y2n - y1n) / TARGET_H
    return [cls, cx_n, cy_n, w_n, h_n]


def augment_images():
    """步骤3: 离线增强, 每张原图生成6张 (含坐标变换)

    原图 | 水平翻转 | 旋转8° | 变亮 | 变暗 | 缩放1.15
    """
    os.makedirs(AUG_IMG, exist_ok=True)
    os.makedirs(AUG_LABEL, exist_ok=True)
    ang = 8
    ang_r = np.deg2rad(ang)

    count = 0
    for f in sorted(os.listdir(RESIZED_IMG)):
        if not f.endswith(".jpg"):
            continue
        base = f[:-4]
        img = cv2.imread(os.path.join(RESIZED_IMG, f))
        H, W = img.shape[:2]
        boxes = read_boxes(os.path.join(RESIZED_LABEL, base + ".txt"))
        if not boxes:
            continue

        # 0: 原图
        cv2.imwrite(os.path.join(AUG_IMG, base + "_0.jpg"), img)
        write_boxes(os.path.join(AUG_LABEL, base + "_0.txt"), boxes)
        count += 1

        # 1: 水平翻转
        img_f = cv2.flip(img, 1)
        boxes_f = [[b[0], 1 - b[1], b[2], b[3], b[4]] for b in boxes]
        cv2.imwrite(os.path.join(AUG_IMG, base + "_1f.jpg"), img_f)
        write_boxes(os.path.join(AUG_LABEL, base + "_1f.txt"), boxes_f)
        count += 1

        # 2: 旋转 8° (框用最小外接矩形, 要放大尺寸)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        img_r = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
        boxes_r = []
        for b in boxes:
            cls, cx, cy, w, h = b
            px, py = (cx - 0.5) * W, (cy - 0.5) * H
            rx = px * np.cos(ang_r) - py * np.sin(ang_r)
            ry = px * np.sin(ang_r) + py * np.cos(ang_r)
            # 关键: 旋转后外接矩形尺寸变大
            new_w = w * W * abs(np.cos(ang_r)) + h * H * abs(np.sin(ang_r))
            new_h = w * W * abs(np.sin(ang_r)) + h * H * abs(np.cos(ang_r))
            boxes_r.append([cls, rx / W + 0.5, ry / H + 0.5, new_w / W, new_h / H])
        cv2.imwrite(os.path.join(AUG_IMG, base + "_2r.jpg"), img_r)
        write_boxes(os.path.join(AUG_LABEL, base + "_2r.txt"), boxes_r)
        count += 1

        # 3: 变亮
        img_b = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
        cv2.imwrite(os.path.join(AUG_IMG, base + "_3b.jpg"), img_b)
        write_boxes(os.path.join(AUG_LABEL, base + "_3b.txt"), boxes)
        count += 1

        # 4: 变暗
        img_d = cv2.convertScaleAbs(img, alpha=0.7, beta=-15)
        cv2.imwrite(os.path.join(AUG_IMG, base + "_4d.jpg"), img_d)
        write_boxes(os.path.join(AUG_LABEL, base + "_4d.txt"), boxes)
        count += 1

        # 5: 缩放 1.15 (中心放大)
        scale = 1.15
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
        img_s = cv2.warpAffine(img, M, (W, H), borderValue=(0, 0, 0))
        boxes_s = []
        for b in boxes:
            cls, cx, cy, w, h = b
            px, py = (cx - 0.5) * W, (cy - 0.5) * H
            nx, ny = px * scale / W + 0.5, py * scale / H + 0.5
            boxes_s.append([cls, nx, ny, w * scale, h * scale])
        cv2.imwrite(os.path.join(AUG_IMG, base + "_5s.jpg"), img_s)
        write_boxes(os.path.join(AUG_LABEL, base + "_5s.txt"), boxes_s)
        count += 1

    print(f"增强完成: {count} 张")


def resize_all():
    """步骤1+2: 批量缩放图片并转换标注"""
    os.makedirs(RESIZED_IMG, exist_ok=True)
    os.makedirs(RESIZED_LABEL, exist_ok=True)
    for f in sorted(os.listdir(SOURCE_IMG)):
        if not f.endswith(".jpg"):
            continue
        base = f[:-4]
        # 读原图尺寸
        img0 = cv2.imread(os.path.join(SOURCE_IMG, f))
        orig_h, orig_w = img0.shape[:2]
        # 缩放
        scale, ox, oy = resize_and_pad(
            os.path.join(SOURCE_IMG, f),
            os.path.join(RESIZED_IMG, f),
        )
        # 转换标注
        txt_src = os.path.join(SOURCE_LABEL, base + ".txt")
        if os.path.exists(txt_src):
            boxes = read_boxes(txt_src)
            new_boxes = [convert_box(b, orig_w, orig_h, scale, ox, oy) for b in boxes]
            write_boxes(os.path.join(RESIZED_LABEL, base + ".txt"), new_boxes)
    print(f"缩放完成: {len(os.listdir(RESIZED_IMG))} 张")


def train():
    """步骤4: 训练 (和 maitrain.py 一致)"""
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        patience=30,
        seed=42,
        verbose=True,
    )
    print("训练完成! best.pt:", results.save_dir)


if __name__ == "__main__":
    print("======== 第1步: 统一图片尺寸 ========")
    resize_all()
    print("======== 第2步: 离线数据增强 ========")
    augment_images()
    print("======== 第3步: 开始训练 ========")
    train()
