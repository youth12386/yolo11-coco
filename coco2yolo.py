import os
import json
from tqdm import tqdm
from PIL import Image

def coco_to_yolo(coco_bbox, img_width, img_height):
    x_min, y_min, width, height = coco_bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return [x_center, y_center, width, height]

def convert_coco_to_yolo(coco_json_path, image_source_dir, labels_output_dir, images_output_dir):
    os.makedirs(labels_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    
    with open(coco_json_path) as f:
        coco_data = json.load(f)
    
    # 创建映射
    image_id_to_info = {img['id']: img for img in coco_data['images']}
    category_id_to_yolo_id = {cat['id']: i for i, cat in enumerate(coco_data['categories'])}
    
    # 按图像分组标注
    image_to_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
        image_to_annotations[image_id].append(ann)
    
    # 处理每张图像
    for image_id, annotations in tqdm(image_to_annotations.items(), desc="Converting"):
        img_info = image_id_to_info[image_id]
        img_filename = img_info['file_name']
        base_filename = os.path.splitext(img_filename)[0]
        
        # 1. 复制图片到images目录
        src_img_path = os.path.join(image_source_dir, img_filename)
        dst_img_path = os.path.join(images_output_dir, img_filename)
        if os.path.exists(src_img_path):
            os.system(f'cp "{src_img_path}" "{dst_img_path}"')
        
        # 2. 创建标注文件到labels目录
        txt_filename = f"{base_filename}.txt"
        txt_path = os.path.join(labels_output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for ann in annotations:
                yolo_class_id = category_id_to_yolo_id[ann['category_id']]
                yolo_bbox = coco_to_yolo(ann['bbox'], img_info['width'], img_info['height'])
                line = f"{yolo_class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n"
                f.write(line)

# 创建标准YOLO目录结构
dataset_root = "yolo_dataset"  # 新的标准目录
os.makedirs(dataset_root, exist_ok=True)

# 转换训练集
convert_coco_to_yolo(
    coco_json_path="dataset/annotations/instances_train2014.json",
    image_source_dir="dataset/train2014",
    labels_output_dir=os.path.join(dataset_root, "labels/train"),
    images_output_dir=os.path.join(dataset_root, "images/train")
)

# 转换验证集
convert_coco_to_yolo(
    coco_json_path="dataset/annotations/instances_val2014.json",
    image_source_dir="dataset/val2014",
    labels_output_dir=os.path.join(dataset_root, "labels/val"),
    images_output_dir=os.path.join(dataset_root, "images/val")
)

print("转换完成！标准YOLO目录已创建在:", dataset_root)