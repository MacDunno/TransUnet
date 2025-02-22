import os  
import cv2  
import numpy as np  

def check_binary_images(folder_path):  
    """  
    检查文件夹下所有图像是否仅包含像素值0和1。  

    Args:  
        folder_path (str): 图像文件夹路径。  

    Returns:  
        None  
    """  
    all_binary = True  # 标志变量，记录是否所有图像都是二值图  
    for filename in os.listdir(folder_path):  
        file_path = os.path.join(folder_path, filename)  

        # 检查是否为图像文件  
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  
            # 读取图像  
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  

            if image is None:  
                print(f"无法读取文件: {filename}")  
                continue  

            # 检查是否仅包含0和1  
            unique_values = np.unique(image)  
            if not np.array_equal(unique_values, [0, 1]):  
                print(f"文件 {filename} 包含非二值像素: {unique_values}")  
                all_binary = False  
            else:  
                print(f"文件 {filename} 是二值图像。")  

    if all_binary:  
        print("文件夹下所有图像均为二值图像（仅包含像素值0和1）。")  
    else:  
        print("文件夹下存在非二值图像。")  

if __name__ == "__main__":  
    # 输入文件夹路径  
    folder_path = "/storage/Code/TransUNet_v1/data/Synapse/ISIC4/train/label"  # 替换为你的文件夹路径  

    check_binary_images(folder_path)