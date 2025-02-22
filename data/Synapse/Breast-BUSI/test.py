import os  
from PIL import Image  

def check_image_label_size(image_folder, label_folder):  
    inconsistent_files = []  

    for image_name in os.listdir(image_folder):  
        if image_name.endswith('.png'):  
            image_path = os.path.join(image_folder, image_name)  
            label_path = os.path.join(label_folder, image_name)  

            if os.path.exists(label_path):  
                with Image.open(image_path) as img:  
                    image_size = img.size  
                with Image.open(label_path) as lbl:  
                    label_size = lbl.size  

                if image_size != label_size:  
                    inconsistent_files.append(image_name)  
            else:  
                print(f"Label for {image_name} not found.")  

    return inconsistent_files  

# 使用示例  
image_folder = '/storage/USFM_TransUNet/data/Synapse/Breast-BUSI/train/img'  
label_folder = '/storage/USFM_TransUNet/data/Synapse/Breast-BUSI/train/label'  
inconsistent_files = check_image_label_size(image_folder, label_folder)  

if inconsistent_files:  
    print("The following files have inconsistent sizes:")  
    for file in inconsistent_files:  
        print(file)  
else:  
    print("All image and label sizes are consistent.")