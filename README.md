# 物品检测入库系统

这是一个基于 YOLO11 + ONNXRuntime + Flask 的物品检测入库系统，用于对拍摄图片中的物品进行识别、定位和分类，并将识别结果返回给上游业务系统。系统支持：

- 物品检测与边界框绘制
- 目标类别识别
- 仓库归类（如食品仓库、家具仓库、日用品仓库等）
- HTTP 接口返回检测结果
- 可接入入库业务流程

## 项目概览

该项目包含以下核心功能：

1. 训练/微调 YOLO11 模型：参考 [train.py](train.py)
2. 基于 ONNX 的推理服务：参考 [app.py](app.py)
3. 图像检测结果输出：返回图片 base64、检测框、标签、分数
4. 物品分类归档：按照仓库类型返回分类结果

这个系统的核心业务逻辑是：

- 摄像头或上传图片作为入库检测输入
- 模型识别出物品类型
- 将识别出的物品归类到对应仓库
- 把结果返回给上层入库管理系统

例如：

- 椅子 -> 家具仓库
- 香蕉 -> 食品仓库
- 书本 -> 日用品仓库
- 交通工具 -> 其他仓库

## 运行环境要求

- Windows 10 / 11 或 Linux
- Miniconda / Anaconda
- Python 3.11
- 可选 NVIDIA GPU（若使用 CUDA）

## 1. 创建 Conda 环境

```powershell
conda create -n yolo11 python=3.11 -y
conda activate yolo11
```

## 2. 安装依赖

推荐直接使用项目中的依赖文件：

```powershell
pip install -r requirements.txt
```

如果你需要手动安装：

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install ultralytics opencv-python-headless flask onnxruntime tqdm numpy Pillow PyYAML
```

## 3. 模型准备

下载权重文件到对应目录中：

[drive.google.com/drive/folders/1Ngzy_qVkACwef0QCBRH2WPK0jf5WuthZ?usp=drive_link](https://drive.google.com/drive/folders/1Ngzy_qVkACwef0QCBRH2WPK0jf5WuthZ?usp=drive_link)

```powershell
cd yolo11-coco

mkdir runs\train\exp\weights
```

## 4. 启动物品检测服务

### 4.1 检测接口

接口地址：

```text
POST /detection
```

请求体示例：

```json
{
  "image_base64": "base64编码的图片数据",
  "model_path": "runs/train/exp/weights/best.onnx"
}
```

返回示例：

```json
{
  "code": "0000",
  "msg": "success",
  "result": {
    "rec_base64": "检测后的图片base64",
    "rec_label": "['person', 'chair']",
    "rec_score": "[0.98, 0.87]",
    "rec_boxes": "[[x1, y1, w, h], ...]",
    "rec_name": ["人", "椅子"],
    "product_label": ["其他仓库", "家具仓库"]
  }
}
```

# 自动化测试说明

本项目中还包含一个自动化测试脚本，用于对 YOLO11 检测服务进行功能、边界值和场景验证，入口文件为 [test_yolo11_detection.py](test_yolo11_detection.py)。该脚本主要用于验证：

- `/detection` 接口是否能正常返回检测结果
- 参数校验是否符合预期
- 仓库分类映射是否正确
- 模型加载失败、图片解码失败等异常场景是否可恢复
- 边界值参数处理是否稳定

## 1 运行方式

在项目根目录执行：

```powershell
conda activate yolo11
python test_yolo11_detection.py
```

运行后脚本会：

1. 自动创建 Flask 测试客户端
2. 构造 base64 图片输入
3. 依次执行 39 个测试用例
4. 输出通过/失败状态
5. 生成测试记录文件 `detection_test_records.csv`

## 2 代码包含的测试模块概览

这份测试脚本的核心逻辑可以概括为：

- 目标检测接口测试：验证 `/detection` 的功能正确性
- 业务分类测试：验证 `get_product_label()` 的仓库归类逻辑
- 异常处理测试：验证服务对错误输入不崩溃
- 数据记录测试：将结果保存为 CSV 便于分析和汇报
