# -*- coding: utf-8 -*-
"""
cam_detect.py —— 摄像头实时目标检测

用预训练 YOLOv8n 模型（COCO 80类）检测摄像头画面里的物体->YOLOv8s
按 q 退出
"""
from ultralytics import YOLO
import cv2

# 加载预训练模型（第一次运行会自动下载 yolov8n.pt，约 6MB）
model = YOLO(r'runs/detect/train-3/weights/best.pt')#记得修改训练模型
'YOLO(模型地址)，r:地址内的反斜杠不会用作转义字符，地址：字符串，从这个代码运行的这一层开始写'
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
    results = model.predict(frame, conf=0.6, verbose=False)
    'model.predict找出目标，frame源，conf可信度，verbose打印日志'
    'result结果是个列表，可用print(type(result))验证，而且长度是1'

    # 把检测结果画到帧上（Ultralytics 自带画框功能）
    annotated = results[0].plot()
    '这个0是编号，编号0对应奥利奥'
    # 显示
    cv2.imshow('YOLO Detect', annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('已退出')
