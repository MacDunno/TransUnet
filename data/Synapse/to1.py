import os  
import cv2  

def replace_pixel_values(folder_path):  
    """  
    将文件夹下所有图像中像素值为255的像素替换为1，并直接覆盖原文件。  

    Args:  
        folder_path (str): 图像文件夹路径。  
    """  
    # 遍历文件夹中的所有文件  
    for filename in os.listdir(folder_path):  
        file_path = os.path.join(folder_path, filename)  

        # 检查是否为图像文件  
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  
            # 读取图像  
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  

            if image is None:  
                print(f"无法读取文件: {filename}")  
                continue  

            # 将像素值为255的像素替换为1  
            image[image >= 128] = 255
            image[image < 128] = 0 

            # 覆盖保存图像  
            cv2.imwrite(file_path, image)  
            print(f"已处理并覆盖: {file_path}")  

if __name__ == "__main__":  
    # 输入文件夹路径  
    folder_path = "/storage/Code/TransUNet_v1/data/Synapse/ISIC5/train/label"  # 替换为你的文件夹路径  

    replace_pixel_values(folder_path)