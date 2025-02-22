import os  
from PIL import Image  

def convert_jpg_to_png_in_place(folder_path):  
    """  
    将文件夹中的所有 JPG 文件直接转换为 PNG 文件，并覆盖原文件  
    :param folder_path: 文件夹路径  
    """  
    # 检查文件夹是否存在  
    if not os.path.exists(folder_path):  
        print(f"文件夹 {folder_path} 不存在，请检查路径")  
        return  

    # 遍历文件夹中所有文件  
    for filename in os.listdir(folder_path):  
        # 检查是否为 JPG 文件  
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):  
            # 获取文件完整路径  
            file_path = os.path.join(folder_path, filename)  
            
            # 生成 PNG 文件路径（替换后缀为 .png）  
            new_file_path = os.path.splitext(file_path)[0] + ".png"  

            try:  
                # 打开 JPG 文件并转换为 RGB 再保存为 PNG  
                with Image.open(file_path) as img:  
                    img = img.convert("RGB")  # 确保转为 RGB 格式  
                    img.save(new_file_path, "PNG")  
                
                # 删除原 JPG 文件  
                os.remove(file_path)  
                print(f"成功将 {filename} 转换为 {os.path.basename(new_file_path)}")  
            except Exception as e:  
                print(f"转换 {filename} 时发生错误：{e}")  

# 示例用法  
if __name__ == "__main__":  
    folder = "/storage/Code/TransUNet_v1/data/Synapse/ISIC5/val/img"  # 替换为你的文件夹路径  
    convert_jpg_to_png_in_place(folder)