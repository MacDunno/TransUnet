from PIL import Image
import numpy as np

def get_unique_pixel_values(image_path):
    # 打开图像
    image = Image.open(image_path)
    
    # 将图像转换为NumPy数组
    image_array = np.array(image)

    # 如果图像是灰度图像，则直接获取唯一像素值
    if len(image_array.shape) == 2:
        unique_values = np.unique(image_array)
    else:
        # 对于彩色图像，获取每个像素的独特组合
        unique_values = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)
    
    return unique_values
# 示例用法
image_path = '/storage/Code/TransUNet_v1/predictions/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_bs24_224/32203804L_pred.png'
unique_pixel_values = get_unique_pixel_values(image_path)

print("图像中不同的像素值：")
for pixel_value in unique_pixel_values:
    print(pixel_value)