import requests
import base64
import cv2
import numpy as np


def base64_to_image(base64_str: str) -> np.ndarray:
    """
    将Base64字符串转换为OpenCV图像

    参数：
        base64_str: 包含可选前缀的Base64编码字符串
                    (如：data:image/jpeg;base64,ABC...)

    返回：
        OpenCV格式的图像数组 (BGR通道顺序)

    异常：
        ValueError: 当输入不是有效的Base64图像数据时
    """
    try:
        # 去除可能的Data URL前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]

        # Base64解码
        image_data = base64.b64decode(base64_str, validate=True)

        # 转换为numpy数组
        nparr = np.frombuffer(image_data, np.uint8)

        # OpenCV解码
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("解码后的数据不是有效的图像格式")

        return img
    except (base64.binascii.Error, TypeError) as e:
        raise ValueError(f"无效的Base64字符串: {str(e)}")
    except Exception as e:
        raise ValueError(f"图像解码失败: {str(e)}")


# test_path = 'dataset/train/images/607-155-1_jpg.rf.e64c13f57c8fcb591afebbfe56ca131d.jpg'
test_path = 'coco-dataset/5.jpg'
with open(test_path, "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()
f = open("test_images/1.txt", 'w')
f.write(img_base64)
f.close()
payload = {
    "model_path": "saved_models/yolo11.onnx",
    "image_base64": img_base64,
    "confidence_thres": 0.5,
    "iou_thres": 0.5
}

response = requests.post(
    "http://localhost:1003/detection",
    json=payload
)
print(response)
print(response.json())
# base64转图片
img = base64_to_image(response.json()['result']['rec_base64'])
cv2.imwrite("service_result.png", img)
