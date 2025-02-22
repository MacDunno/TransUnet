
import os

def list_filenames_without_extension(folder_path, output_file):
    # 确保文件夹路径存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for file in files:
            # 检查是否为文件（不是文件夹）
            if os.path.isfile(os.path.join(folder_path, file)):
                # 获取文件名（不包括扩展名）
                filename_without_extension = os.path.splitext(file)[0]
                # 写入文件名到输出文件
                f.write(filename_without_extension + '\n')

    print(f"文件名已保存到 {output_file}")

# 使用示例
folder_path = 'data/Synapse/1617D/test_only/img'  # 替换为您要处理的文件夹路径
output_file = 'lists/lists_Synapse/1617D/test_only.txt'  # 输出文件名

list_filenames_without_extension(folder_path, output_file)
