import json
import os
import threading
import time
from typing import Union
import onnxruntime
import cv2
import numpy
import numpy as np


class YOLOv11(object):
    """使用yolo的.pt格式模型转换为.onnx格式模型进行目标检测"""

    input_types_table = {'tensor(float)': numpy.float32, 'tensor(float16)': numpy.float16}

    def __init__(self, **kwargs):
        self.initConfig(**kwargs)

    def initModel(self, path, t: str = None) -> None:
        """
        初始化模型
        :param path: model path
        :param t: onnxruntime or cv2.dnn, default: onnxruntime
        :return:
        """

        self.net = onnxruntime.InferenceSession(path)
        model_inputs = self.net.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_types = [model_inputs[i].type for i in range(len(model_inputs))]

        model_outputs = self.net.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        self.has_postprocess = 'score' in self.output_names

    def initConfig(self, input_width=640, input_height=640, conf_thres=0.5, iou_thres=0.5, draw_box=True,
                   thickness=2, class_names: Union[tuple, list] = None,
                   box_color=(255, 0, 0),
                   txt_color=(0, 255, 255), with_pos=False,
                   **kwargs) -> None:
        """
        初始化模型配置
        """
        assert 0 < conf_thres <= 1 and 0 < iou_thres <= 1
        self.input_width = input_width  # 输入图片宽
        self.input_height = input_height  # 输入图片高
        self.conf_threshold = conf_thres  # 置信度
        self.iou_threshold = iou_thres  # IOU
        self.draw_box = draw_box  # 是否画锚框
        self.thickness = thickness  # 锚框宽度
        self.class_names = class_names  # 类别
        self.box_color = box_color  # 锚框颜色 BGR
        self.txt_color = txt_color  # 文字颜色 BGR
        self.with_pos = with_pos  # 是否返回坐标
        self.__dict__.update(kwargs)

    def detect(self, image: numpy.ndarray) -> dict:
        outputs = self.inference(image)
        boxes, scores, class_ids = self.__postProcess(outputs)
        res_ = self.__formatResult(image, boxes, scores, class_ids)
        return res_

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        # print(f"Original image shape: {shape}")

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
        # print(f"Final letterboxed image shape: {img.shape}")
        img=cv2.resize(img,(640,640))

        return img, (r, r), (dw, dh)

    def __prepareInput(self, image):
        self.img_height, self.img_width = image.shape[:2]
        # 将图像颜色空间从 BGR 转换为 RGB
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img=image
        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))

        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0
        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def inference(self, image: numpy.ndarray) -> list[numpy.ndarray]:
        """
        :param image: 待检测图像 RGB格式
        :return:
        """
        input_tensor = self.__prepareInput(image)
        # blob = cv2.dnn.blobFromImage(input_img, 1 / 255.0)
        # Perform inference on the image
        outputs = self.net.run(self.output_names, {self.input_names[0]: input_tensor})
        # print(outputs[0].shape)  # (1, 25200, 85)
        return outputs

    def __postProcess(self, outputs):
        if self.has_postprocess:
            res_ = self.__parseProcessedOutput(outputs)
        else:
            # Process output data
            res_ = self.processOutput(outputs)
        return res_

    def processOutput(self, output) -> tuple:
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []

        # 计算缩放比例和填充
        ratio = self.img_width / self.input_width, self.img_height / self.input_height

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.conf_threshold:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw  # 移除填充
                y -= self.dh
                x /= self.ratio[0]  # 缩放回原图
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = x - w / 2
                top = y - h / 2
                right = x + w / 2
                bottom = y + h / 2
                width = int(w)
                height = int(h)

                boxes.append([left, top, right, bottom])
                scores.append(max_score)
                class_ids.append(class_id)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = np.array(list(
            map(lambda x: int(x), cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)))).flatten()
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        if indices.any():
            return boxes[indices], scores[indices], class_ids[indices]
        else:
            return None, None, None

    def __parseProcessedOutput(self, outputs):
        scores = numpy.squeeze(outputs[self.output_names.index('score')])
        predictions = outputs[self.output_names.index('batchno_classid_x1y1x2y2')]

        # Filter out object scores below threshold
        valid_scores = scores > self.conf_threshold
        predictions = predictions[valid_scores, :]
        scores = scores[valid_scores]

        # Extract the boxes and class ids
        # batch_number = predictions[:, 0]
        class_ids = predictions[:, 1]
        boxes = predictions[:, 2:]

        # In postprocess, the x,y are the y,x
        boxes = boxes[:, [1, 0, 3, 2]]

        # Rescale boxes to original image dimensions
        boxes = self.__rescaleBoxes(boxes)

        return boxes, scores, class_ids

    def __extractBoxes(self, predictions):
        boxes = predictions[:, :4]  # Extract boxes from predictions
        boxes = self.__rescaleBoxes(boxes)  # Scale boxes to original image dimensions
        # Convert boxes to xyxy format
        boxes_ = numpy.copy(boxes)
        boxes_[..., 0] = boxes[..., 0] - boxes[..., 2] * 0.5
        boxes_[..., 1] = boxes[..., 1] - boxes[..., 3] * 0.5
        boxes_[..., 2] = boxes[..., 0] + boxes[..., 2] * 0.5
        boxes_[..., 3] = boxes[..., 1] + boxes[..., 3] * 0.5
        return boxes_

    def __rescaleBoxes(self, boxes):
        """Rescale boxes to original image dimensions"""
        input_shape = numpy.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = numpy.divide(boxes, input_shape, dtype=numpy.float32)
        boxes *= numpy.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def __formatResult(self, image, boxes, scores, class_ids) -> dict:
        """
        格式化检测结果
        :param image: 输入图像
        :param boxes:
        :param scores:
        :param class_ids:
        :return: 检测结果
        """
        detection = {}
        if boxes is None:
            return detection
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            label = self.class_names[class_id]
            if label == '_':
                continue
            if label in detection:
                detection[label]['数量'] += 1
                #detection[label]['score'].append(score)
                if self.with_pos: detection[label]['pos'].append((x1, y1, x2, y2))
            else:
                detection[label] = {'数量': 1, }
                if self.with_pos: detection[label].update({'pos': [(x1, y1, x2, y2), ]})

            if self.draw_box:
                label_p = f'{label} {int(score * 100)}%'
                lw = max(round(sum(image.shape) / 2 * 0.003), self.thickness)  # line width
                cv2.rectangle(image, (x1, y1), (x2, y2), self.box_color, thickness=lw, lineType=cv2.LINE_AA)
                if label_p:
                    tf = max(lw - 1, 1)  # font thickness
                    w, h = cv2.getTextSize(label_p, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
                    outside = y1 - h >= 3
                    p2 = x1 + w, y1 - h - 3 if outside else y1 + h + 3
                    cv2.rectangle(image, (x1, y1), p2, self.box_color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(image,
                                label_p, (x1, y1 - 2 if outside else y1 + h + 2),
                                0,
                                lw / 3,
                                self.txt_color,
                                thickness=tf,
                                lineType=cv2.LINE_AA)
        return detection


class DataLoader(object):
    """逐帧加载图像，返回RGB格式"""
    VIDEO_TYPE = ('asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv')
    IMAGE_TYPE = ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')
    URL_TYPE = ('rtsp://', 'rtmp://', 'http://', 'https://')

    def __init__(self, source: Union[int, str], frame_skip=-1, flip=None, rotate=None, **kwargs):
        """
        :param source: input source
        :param frame_skip: frame skip or not, <0: auto; =0: dont skip; >0: skip  # video only
        :param flip:
        """
        self.source, *self.params = str(source).split()
        self.flip = flip
        self.rotate = rotate
        self.is_wabcam = self.source.isnumeric()
        self.is_video = self.source.lower().endswith(DataLoader.VIDEO_TYPE)
        self.is_image = self.source.lower().endswith(DataLoader.IMAGE_TYPE)
        self.is_screen = self.source.startswith('screen')
        self.is_url = self.source.lower().startswith(DataLoader.URL_TYPE)
        assert self.is_wabcam or self.is_video or self.is_image or self.is_screen or self.is_url, \
            f'Invalid or unsupported file format: {self.source}'

        if self.is_wabcam:
            self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.cap.isOpened(), f'Failed to load: {self.source}'
        elif self.is_video or self.is_image or self.is_url:
            self.frame_skip = frame_skip if not self.is_image else 0
            self.idx = 0
            if self.frame_skip < 0:
                class VideoFrameDraw(threading.Thread):
                    def __init__(self):
                        super(VideoFrameDraw, self).__init__(daemon=True)
                        self.cap = cv2.VideoCapture(source)
                        self.grab = self.cap.grab
                        self.isOpened = self.cap.isOpened
                        self.release = self.cap.release
                        assert self.cap.isOpened(), f'Failed to load {source}'

                        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        self.w, self.h = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                        self.ret = self.cap.grab()
                        if not self.ret:
                            return
                        self.ret, self.img = self.cap.retrieve()
                        if not self.ret or self.img is None:
                            return
                        self.pause = threading.Event()
                        self.read_frame = False

                        self.start()

                    def retrieve(self):
                        if not self.pause.isSet(): self.pause.set()
                        if self.ret:
                            self.read_frame = True
                            return self.ret, self.img
                        return False, None

                    def run(self) -> None:
                        while self.cap.isOpened():
                            self.ret = self.cap.grab()
                            self.pause.wait()
                            if not self.ret:
                                break
                            time.sleep(self.fps / 1000)
                            if not self.read_frame:
                                continue
                            self.ret, self.img = self.cap.retrieve()
                            self.read_frame = False
                            if not self.ret:
                                break

                    def __del__(self):
                        self.release()

                self.cap = VideoFrameDraw()
                self.w, self.h = int(self.cap.w), int(self.cap.h)
                self.fps = self.cap.fps
            else:  # frame_skip >= 0
                self.cap = cv2.VideoCapture(self.source)
                self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            assert self.cap.isOpened(), f'Failed to load: {self.source}'
        elif self.is_screen:
            import mss
            screen, left, top, width, height = 0, None, None, None, None  # default to full screen 0
            if len(self.params) == 1:
                screen = int(self.params[0])
            elif len(self.params) == 4:
                left, top, width, height = (int(x) for x in self.params)
            elif len(self.params) == 5:
                screen, left, top, width, height = (int(x) for x in self.params)
            self.sct = mss.mss()

            # Parse monitor shape
            monitor = self.sct.monitors[screen]
            top = monitor["top"] if top is None else (monitor["top"] + top)
            left = monitor["left"] if left is None else (monitor["left"] + left)
            width = width or monitor["width"]
            height = height or monitor["height"]
            self.monitor = {"left": left, "top": top, "width": width, "height": height}

    def __next__(self) -> tuple[numpy.ndarray, str]:
        if self.is_wabcam:
            ret, img = self.cap.read()
            path = ''
        elif self.is_video or self.is_image or self.is_url:
            while self.idx <= self.frame_skip:
                ret = self.cap.grab()
                self.idx += 1
                if not ret:
                    raise StopIteration
            ret, img = self.cap.retrieve()
            self.idx = 0
            path = self.source
        elif self.is_screen:
            ret = True
            img = numpy.array(self.sct.grab(self.monitor))[:, :, :3]
            path = ''
        else:
            raise StopIteration

        if not ret or img is None:
            raise StopIteration
        if self.flip is not None: img = cv2.flip(img, self.flip)
        if self.rotate is not None: img = cv2.rotate(img, self.rotate)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, os.path.basename(path)

    def __iter__(self):
        return self

    def __del__(self):
        if 'cap' in self.__dict__:
            self.cap.release()
