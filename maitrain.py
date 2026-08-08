from ultralytics import YOLO

#加载预训练模型 yolov8n
model = YOLO("yolov8n.pt")
'''
yolov8n.pt：n = nano，最轻量版本，速度快，适合机器人部署。
pt 文件：已经在 COCO 大型数据集预训练好的权重。
迁移学习：不是从零训练网络，在别人训练好的基础上，再学你的奥利奥样本，小数据集也能收敛。

'''
#训练
results = model.train(

data = r'C:\Users\lhl87\Desktop\maiVision\maidata.yaml'
, #数据配置 ，指定数据集 yaml 配置文件，告诉 YOLO 数据集路径、类别名字。
    epochs = 100,   # 训练轮数
    imgsz = 640,    # 输入图片尺寸
    batch = 16, # 每批几张  每一次同时喂给 GPU16 张图片
    device = 0,    #使用0号显卡训练
    patience=30,    #早停机制：连续 30 个 epoch，验证集指标没有提升，直接终止训练，防止过拟合。
    seed=42,    #随机种子，固定随机划分、数据增强，保证每次训练结果可以复现。
verbose=True,   #打印详细日志，每一轮输出损失、mAP、召回率，方便看哪里出问题

)

#验证
model.val()

# 4. 保存最佳模型
print('训练完成！模型保存在 runs/detect/train/weights/best.pt')