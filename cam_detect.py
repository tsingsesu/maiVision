# -*- coding: utf-8 -*-
"""
cam_detect.py —— 摄像头实时目标检测

用预训练 YOLOv8n 模型（COCO 80类）检测摄像头画面里的物体
按 q 退出
"""
from ultralytics import YOLO
import cv2

# 加载预训练模型（第一次运行会自动下载 yolov8n.pt，约 6MB）
model = YOLO('yolov8n.pt')

# 打开摄像头（0 = 内置摄像头）
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print('按 q 退出 | 对着镜头挥挥手或放个瓶子试试')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推理：把当前帧送给 YOLO，得到检测结果
    results = model.predict(frame, conf=0.4, verbose=False)

    # 把检测结果画到帧上（Ultralytics 自带画框功能）
    annotated = results[0].plot()

    # 显示
    cv2.imshow('YOLO Detect', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('已退出')
