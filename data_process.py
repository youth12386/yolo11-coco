import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil

# 定义XML标注文件夹和图片文件夹路径
xml_folder = 'insulator/Annotations'
image_folder = 'insulator/JPEGImages'

# 定义输出根文件夹
output_root = 'yolo_data'
os.makedirs(output_root, exist_ok=True)

# 定义类别列表
classes = ['insulator']

# 创建子文件夹
image_train_folder = os.path.join(output_root, 'images', 'train')
image_val_folder = os.path.join(output_root, 'images', 'val')
image_test_folder = os.path.join(output_root, 'images', 'test')
label_train_folder = os.path.join(output_root, 'labels', 'train')
label_val_folder = os.path.join(output_root, 'labels', 'val')
label_test_folder = os.path.join(output_root, 'labels', 'test')

for folder in [image_train_folder, image_val_folder, image_test_folder,
               label_train_folder, label_val_folder, label_test_folder]:
    os.makedirs(folder, exist_ok=True)


def convert_xml_to_yolo(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取图片尺寸
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        class_id = classes.index(class_name)

        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # 计算YOLO格式的边界框坐标
        x_center = (xmin + xmax) / (2 * width)
        y_center = (ymin + ymax) / (2 * height)
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height

        yolo_lines.append(f"{class_id} {x_center} {y_center} {w} {h}")

    return yolo_lines


# 遍历XML文件夹，转换为YOLO格式
xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 拆分数据为训练集和验证集
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
test_files = train_files  # 测试集和训练集使用同一批数据

# 处理训练集
for file in train_files:
    base_name = os.path.splitext(os.path.basename(file))[0]
    xml_file = os.path.join(xml_folder, f"{base_name}.xml")
    yolo_lines = convert_xml_to_yolo(xml_file)
    yolo_file = os.path.join(label_train_folder, f"{base_name}.txt")
    with open(yolo_file, 'w') as f:
        for line in yolo_lines:
            f.write(line + '\n')
    shutil.copy(file, os.path.join(image_train_folder, os.path.basename(file)))

# 处理验证集
for file in val_files:
    base_name = os.path.splitext(os.path.basename(file))[0]
    xml_file = os.path.join(xml_folder, f"{base_name}.xml")
    yolo_lines = convert_xml_to_yolo(xml_file)
    yolo_file = os.path.join(label_val_folder, f"{base_name}.txt")
    with open(yolo_file, 'w') as f:
        for line in yolo_lines:
            f.write(line + '\n')
    shutil.copy(file, os.path.join(image_val_folder, os.path.basename(file)))

# 处理测试集
for file in test_files:
    base_name = os.path.splitext(os.path.basename(file))[0]
    xml_file = os.path.join(xml_folder, f"{base_name}.xml")
    yolo_lines = convert_xml_to_yolo(xml_file)
    yolo_file = os.path.join(label_test_folder, f"{base_name}.txt")
    with open(yolo_file, 'w') as f:
        for line in yolo_lines:
            f.write(line + '\n')
    shutil.copy(file, os.path.join(image_test_folder, os.path.basename(file)))

print("数据转换和拆分完成。")