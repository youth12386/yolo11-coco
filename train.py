# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics/cfg/models/11/yolo11.yaml')
    model.load("weights/yolo11m.pt")
    model.train(data=r'yolo_dataset/data.yaml',
                imgsz=640,
                epochs=10,
                batch=128,
                workers=0,
                device='0',
                optimizer='SGD',
                close_mosaic=10,
                resume=False,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
