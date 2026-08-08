# pipeline 使用方法

> 用途：把 Label Studio 标注好的图片，一步步变成 YOLO 训练集，然后训练。
> 对应文件：`pipeline.py`（数据准备流水线）+ `maitrain.py`（训练脚本）+ `maidata.yaml`（数据配置）

---

## 一、总体流程（先看清全貌）

```
拍照片 → Label Studio 标注 → 导出 → 整理到 yolo_train → pipeline.py(缩放+增强) → 训练
   ①          ②            ③         ④                ⑤                   ⑥
```

本手册从 **③ 导出之后** 讲起（① ② 见《标注工具操作》）。

---

## 二、第 3 步：从 Label Studio 导出

在 Label Studio 里标注完所有图片后：

1. 右上角点 **Export**（导出）
2. 格式选 **YOLO**（导出会生成 `classes.txt` + `labels/xxx.txt`）
3. **务必勾选「连同图片一起导出」**——否则导出的只有标注 txt，没有图片（后面会配对失败）

**导出得到的文件夹结构**（在桌面，形如 `project-1-at-2026-08-08-15-18-b2e878e0`）：
```
project-xxx/
├── classes.txt          # 类别列表, 一行一个英文类名
├── images/              # 图片 (文件名带 hash 前缀, 如 01d3cd5c-aoliao_17.jpg)
└── labels/              # 标注 (与图片同名 .txt)
```

**注意**：图片文件名可能是 `hash前缀-实际名.jpg`（如 `01d3cd5c-aoliao_17.jpg`）。pipeline 会自动去掉 hash 前缀取 `aoliao_17`，但你整理时要留意。

---

## 三、第 4 步：整理到 yolo_train（关键！）

### 3.1 把图片放进 train 目录

把导出的**图片**复制到：
```
yolo_train/images/train/
```

建议重命名成有意义的英文名（如 `aoliao_01.jpg`、`shupian_01.jpg`）。**文件名就是你的标签 ID 依据**，要规范。

### 3.2 把标注 txt 放进 labels 目录（改名为与图片一致）

把导出的 `labels/xxx.txt` 复制到：
```
yolo_train/labels/train/
```

**关键：txt 文件名必须和图片文件名完全一致**（去掉 hash 前缀）。例如：
- 图片：`yolo_train/images/train/aoliao_17.jpg`
- 标注：`yolo_train/labels/train/aoliao_17.txt`

**为什么**：YOLO 靠「同名配对」把图片和标注关联起来。文件名对不上，训练会报「图片没有标注」或静默跳过。

### 3.3 验证配对

跑这个检查（确认图片和 txt 一一对应）：
```bash
python -c "
import os
imgs = set(os.path.splitext(f)[0] for f in os.listdir('images/train'))
txts = set(os.path.splitext(f)[0] for f in os.listdir('labels/train'))
print('有图无txt:', imgs - txts or '无')
print('有txt无图:', txts - imgs or '无')
"
```

---

## 四、第 5 步：运行 pipeline.py（缩放 + 增强）

### 4.1 先改配置区

打开 `pipeline.py` 顶部的配置区，确认路径指向你的数据：
```python
SOURCE_IMG = r"C:\Users\lhl87\Desktop\yolo_train\images\train"      # 原始图片
SOURCE_LABEL = r"C:\Users\lhl87\Desktop\yolo_train\labels\train"    # 原始标注
```
> 如果你的数据在别的目录，改这里。下面的 `RESIZED_*`、`AUG_*` 是处理后的输出目录，一般不用动。

### 4.2 确认 data.yaml

`maidata.yaml` 里的 `train` 必须指向 **pipeline 增强后的目录**：
```yaml
path : C:\Users\lhl87\Desktop\yolo_train
train : images/train_aug     # 指向增强后的 180 张
val : images/val             # 验证集 (官方 5 张)
names:
  0: aoliao_heiqiao
```

### 4.3 运行

```bash
cd C:\Users\lhl87\Desktop\maiVision
python pipeline.py
```

pipeline 会自动：
1. **统一图片尺寸**：所有图缩放到 1280×720（长边适配 + 黑边居中），标注坐标同步转换
2. **离线增强**：每张图生成 6 张（原图/翻转/旋转/变亮/变暗/缩放），共 30×6=180 张，标注坐标自动跟着变
3. **启动训练**

---

## 五、第 6 步：训练（如果不想用 pipeline 里的训练，单独跑）

```bash
python maitrain.py
```

训练参数都在 `maitrain.py` 里（epochs=100, imgsz=640, batch=16, device=0…）。

训练完成后，最佳模型在：
```
runs/detect/train_vX/weights/best.pt
```
（pipeline 里指定了 `name='train_v5'` 这类版本号，每次训练会生成新目录，不会覆盖旧的。）

---

## 六、验证结果

```python
from ultralytics import YOLO
model = YOLO(r'runs/detect/train_v5/weights/best.pt')
results = model.predict(r'yolo_train/images/val/角度1.jpg', conf=0.25, verbose=False)
print(results[0].boxes)   # 看框和置信度
```

**期望**：验证集（官方图）上能框出奥利奥，置信度 > 0.8。

---

## 七、常见坑（亲测踩过）

| 坑 | 现象 | 解决 |
|---|---|---|
| **txt 和图片没配对** | 训练报错或 mAP 异常 | 严格按第 3.2 节同名配对 |
| **图片尺寸不统一** | 模型学不会（我们踩过） | 用 pipeline 的缩放，统一 1280×720 |
| **数据太少过拟合** | 训练集 mAP 高、新图不识别 | 用 pipeline 的增强，30→180 张 |
| **背景单一** | 模型记背景不记物体 | 换背景拍多样照片 |
| **只标 1 个类** | 分类置信度低 | 加类别数据，或后期调阈值 |
| **export 没带图片** | 只有 txt 没图 | 导出时勾选「连同图片」 |

---

## 八、以后加新类（如薯片、魔爪）的流程

1. 拍薯片照片 → Label Studio 标注（标签填 `shupian_qingning`）
2. 导出 → 图片放进 `images/train`，txt 放进 `labels/train`，**文件名配对**
3. 更新 `maidata.yaml` 的 `names` 加入新类：
   ```yaml
   names:
     0: aoliao_heiqiao
     1: shupian_qingning   # 新增
   ```
4. 跑 `python pipeline.py` → 训练 → 验证

**关键**：新增类只能**追加到 names 末尾**，不能插队——因为类名顺序 = 类别 ID，改了顺序已标注的数据全乱。

---

## 九、一句话总结

> **Label Studio 导出 → 图片和 txt 同名配对放进 yolo_train → 跑 pipeline.py（缩放+增强）→ 训练 → 验证。**
