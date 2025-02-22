import numpy as np
from PIL import Image
import os

# 加载npz文件
data = np.load(r'/storage/Code/TransUNet/data/Synapse/test_vol_h5/test71.npz', allow_pickle=True)
image_array, label_array = data['image'], data['label']

# 创建输出目录
output_dir = r'/storage/Code/TransUNet/data/Synapse/output_images'
os.makedirs(output_dir, exist_ok=True)

# 将image数组转换并保存为PNG
image = Image.fromarray(image_array)
image.save(os.path.join(output_dir, 'image.png'))

# 如果label是二值图像(0和255)或者灰度图像，也可以直接保存
# 如果是多通道或其他格式，根据需要适当转换
label = Image.fromarray(label_array)
label.save(os.path.join(output_dir, 'label.png'))

print("Images saved successfully!")