from ultralytics import YOLO


def main():
    # 加载预训练模型。现在是 yolov8s（比 nano 大一号，特征更强）
    # yolov8n.pt：n = nano，最轻量，速度快，适合机器人部署。
    # yolov8s.pt：s = small，特征更强，对"奥利奥vs杂物"的区分更好。
    # pt 文件：在 COCO 大型数据集预训练好的权重。
    # 迁移学习：在别人训练好的基础上再学你的奥利奥样本，小数据集也能收敛。
    model = YOLO("yolov8s.pt")

    # 训练
    results = model.train(
        data=r'C:\Users\lhl87\Desktop\maiVision\maidata.yaml',  # 数据配置
        epochs=100,     # 训练轮数
        imgsz=640,      # 输入图片尺寸
        batch=16,       # 每批几张，同时喂给 GPU
        device=0,       # 使用0号显卡
        patience=30,    # 早停：连续30轮没提升就终止
        seed=42,        # 随机种子，保证结果可复现
        verbose=True,   # 打印详细日志
    )

    # 验证
    model.val()
    print('训练完成！模型保存在 runs/detect/train/weights/best.pt')


if __name__ == '__main__':
    # Windows 上必须加这层保护，否则 DataLoader 多进程启动会报错
    main()