import os  
import shutil  
import random  

def create_dir_structure(base_dir):  
    """Create directory structure for train, val, and test sets."""  
    for split in ['train', 'val', 'test']:  
        img_dir = os.path.join(base_dir, split, 'img')  
        label_dir = os.path.join(base_dir, split, 'label')  
        os.makedirs(img_dir, exist_ok=True)  
        os.makedirs(label_dir, exist_ok=True)  

def split_dataset(img_dir, label_dir, base_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):  
    """Split dataset into train, val, and test sets."""  
    img_files = sorted(os.listdir(img_dir))  
    label_files = sorted(os.listdir(label_dir))  

    # Ensure the number of images and labels match  
    assert len(img_files) == len(label_files), "Mismatch between image and label files."  

    # Shuffle the dataset  
    combined = list(zip(img_files, label_files))  
    random.shuffle(combined)  
    img_files[:], label_files[:] = zip(*combined)  

    # Calculate split indices  
    total_count = len(img_files)  
    train_count = int(total_count * train_ratio)  
    val_count = int(total_count * val_ratio)  

    # Split the dataset  
    train_files = combined[:train_count]  
    val_files = combined[train_count:train_count + val_count]  
    test_files = combined[train_count + val_count:]  

    # Function to copy files to the respective directories  
    def copy_files(file_list, split):  
        for img_file, label_file in file_list:  
            shutil.copy(os.path.join(img_dir, img_file), os.path.join(base_dir, split, 'img', img_file))  
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(base_dir, split, 'label', label_file))  

    # Copy files to train, val, and test directories  
    copy_files(train_files, 'train')  
    copy_files(val_files, 'val')  
    copy_files(test_files, 'test')  

if __name__ == "__main__":  
    # Define directories  
    base_dir = 'data/Synapse/1617D'  
    img_dir = os.path.join(base_dir, 'img')  
    label_dir = os.path.join(base_dir, 'label')  

    # Create directory structure  
    create_dir_structure(base_dir)  

    # Split the dataset  
    split_dataset(img_dir, label_dir, base_dir)