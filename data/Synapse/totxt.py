import os  

def write_filenames_to_txt(folder_path, output_txt):  
    """  
    将指定文件夹下的所有文件名（不包含后缀）写入到一个 TXT 文件。  
    :param folder_path: 文件夹路径  
    :param output_txt: 输出的 TXT 文件路径  
    """  
    # 检查文件夹是否存在  
    if not os.path.exists(folder_path):  
        print(f"文件夹 {folder_path} 不存在，请检查路径")  
        return  
    
    # 获取文件夹中的所有文件名并去掉后缀  
    filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path)]  

    # 将文件名逐行写入到 TXT 文件  
    try:  
        with open(output_txt, "w", encoding="utf-8") as f:  
            for filename in filenames:  
                f.write(filename + "\n")  
        print(f"文件夹 {folder_path} 的文件名（不含后缀）已写入到 {output_txt}")  
    except Exception as e:  
        print(f"写入过程中发生错误：{e}") 

# 示例用法  
if __name__ == "__main__":  
    folder = "/storage/Code/TransUNet_v1/data/Synapse/1617D/1617d/img"  # 替换为文件夹路径  
    output = "/storage/Code/TransUNet_v1/lists/lists_Synapse/1617D/1617d.txt"  # 替换为输出 TXT 的路径  
    write_filenames_to_txt(folder, output)