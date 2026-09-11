import argparse
import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
from flask import Flask, request, jsonify
from datetime import datetime

# 类外定义类别映射关系，使用字典格式
CLASS_NAMES =  ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


chinese_names =["人", "自行车", "汽车", "摩托车", "飞机", "公共汽车", "火车", "卡车", "船", "交通信号灯", "消防栓", "停车标志", "停车计时器", "长凳", "鸟", "猫", "狗", "马", "羊", "牛", "大象", "熊", "斑马", "长颈鹿", "背包", "雨伞", "手提包", "领带", "行李箱", "飞盘", "滑雪板", "单板滑雪板", "运动球", "风筝", "棒球棒", "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子", "酒杯", "杯子", "叉子", "刀", "勺子", "碗", "香蕉", "苹果", "三明治", "橙子", "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈", "蛋糕", "椅子", "沙发", "盆栽植物", "床", "餐桌", "马桶", "电视", "笔记本电脑", "鼠标", "遥控器", "键盘", "手机", "微波炉", "烤箱", "烤面包机", "水槽", "冰箱", "书", "时钟", "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"]


class YOLO11:
    """YOLO11 目标检测模型类，用于处理推理和可视化。"""

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        初始化 YOLO11 类的实例。
        参数：
            onnx_model: ONNX 模型的路径。
            input_image: 输入图像的路径。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # 加载类别名称
        self.classes = CLASS_NAMES

        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def preprocess(self):
        """
        对输入图像进行预处理，以便进行推理。
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        # 将 base64 编码的图像解码为字节流
        img_bytes = base64.b64decode(self.input_image)
        # 将字节流转换为 numpy 数组
        img_array = np.frombuffer(img_bytes, np.uint8)
        # 使用 OpenCV 解码图像
        self.img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = self.img.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))

        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0

        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # 返回预处理后的图像数据
        return image_data

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        print(f"Original image shape: {shape}")

        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)

        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2

        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        img = cv2.resize(img, (640, 640))
        print(f"Final letterboxed image shape: {img.shape}")

        return img, (r, r), (dw, dh)

    def postprocess(self, input_image, output):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        参数：
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        # 计算缩放比例和填充
        ratio = self.img_width / self.input_width, self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw  # 移除填充
                y -= self.dh
                x /= self.ratio[0]  # 缩放回原图
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)

                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        result_labels = []
        result_scores = []
        bboxes = []
        for i in indices:
            box = boxes[i]
            bboxes.append(box)
            score = scores[i]
            class_id = class_ids[i]
            result_labels.append(self.classes[class_id])
            if self.classes[class_id] not in result_scores or score > result_scores[self.classes[class_id]]:
                result_scores.append(score)
            self.draw_detections(input_image, box, score, class_id)
        return input_image, result_labels, result_scores, bboxes

    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        参数：
            img: 用于绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测分数。
            class_id: 检测到的目标类别 ID。

        返回：
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def main(self):
        # 使用 ONNX 模型创建推理会话，自动选择 CPU 或 GPU
        session = ort.InferenceSession(
            self.onnx_model,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else [
                "CPUExecutionProvider"],
        )
        # 打印模型的输入尺寸
        print("YOLO11 🚀 目标检测 ONNXRuntime")
        print("模型名称：", self.onnx_model)

        # 获取模型的输入形状
        model_inputs = session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height}")

        # 预处理图像数据，确保使用模型要求的尺寸 (640x640)
        img_data = self.preprocess()

        # 使用预处理后的图像数据运行推理
        outputs = session.run(None, {model_inputs[0].name: img_data})
        print(outputs[0].shape)
        # 对输出进行后处理以获取输出图像
        output_image, result_labels, result_scores, boxes = self.postprocess(self.img, outputs)
        return output_image, result_labels, result_scores, boxes


app = Flask(__name__)

# 定义仓库分类映射
warehouse_categories = {
    "其他仓库": [
        "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
    ],
    "体育用品仓库": [
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket"
    ],
    "食品仓库": [
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake"
    ],
    "家具仓库": [
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator"
    ],
    "日用品仓库": [
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "backpack",
        "umbrella", "handbag", "tie", "suitcase"
    ]
}

def get_product_label(label):
    """根据英文标签获取对应的仓库分类"""
    for warehouse, items in warehouse_categories.items():
        if label in items:
            return warehouse
    return "未知仓库"  # 默认分类（理论上所有标签都应被分类）

@app.route('/detection', methods=['POST'])
def pole_detection():
    data = request.get_json()
    base64_image = data.get('image_base64')
    onnx_model_path = data.get('model_path')
    conf_thres = 0.5
    iou_thres = 0.5

    if base64_image is None or onnx_model_path is None:
        return jsonify({
            'code': '0001',
            'log_id': datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            'msg': 'Missing base64_image or onnx_model_path',
            'result': {}
        })

    try:
        detection = YOLO11(onnx_model_path, base64_image, conf_thres, iou_thres)
        output_image, result_labels, result_scores, boxes = detection.main()

        # 将输出图像转换为 base64 编码
        _, buffer = cv2.imencode('.jpg', output_image)
        output_base64 = base64.b64encode(buffer).decode('utf-8')

        # 转换为中文名称和仓库分类
        chinese_result = [chinese_names[CLASS_NAMES.index(label)] for label in result_labels]
        product_labels = [get_product_label(label) for label in result_labels]

        return jsonify({
            'code': '0000',
            'log_id': datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            'msg': 'success',
            'result': {
                'rec_base64': output_base64,
                'rec_label': str(result_labels),
                'rec_score': str(result_scores),
                'rec_boxes': str(boxes),
                'rec_name': chinese_result,
                'product_label': product_labels  # 新增分类结果字段
            }
        })
    except Exception as e:
        return jsonify({
            'code': '0002',
            'log_id': datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            'msg': f'Error: {str(e)}',
            'result': {}
        })
if __name__ == "__main__":
    app.run(debug=True, port=1003,host='0.0.0.0')
