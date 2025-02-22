import os
import numpy as np
import cv2

def convert_image_to_grayscale(image):
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def convert_label_pixel_values(label):
    label[label == 255] = 1
    return label

def process_npz_file(file_path):
    # 加载数据
    data = np.load(file_path)
    image, label = data['image'], data['label']
    
    # 转换图像为灰度
    image = convert_image_to_grayscale(image)
    
    # 转换标签的像素值
    label = convert_label_pixel_values(label)
    
    # 保存修改后的数据，覆盖原有文件或另存为新文件
    np.savez(file_path, image=image, label=label)

def process_all_npz_files_in_directory(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.npz'):
            file_path = os.path.join(directory_path, file_name)
            process_npz_file(file_path)
            print(f"Processed {file_name}")

# 使用示例
directory_path = '/storage/Code/TransUNet/data/Synapse/train_npz'
process_all_npz_files_in_directory(directory_path)