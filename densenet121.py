import torch  
import torch.nn as nn  
import torch.optim as optim  
from torchvision import datasets, transforms  
from torch.utils.data import Dataset, DataLoader  
import timm  
import pandas as pd  
import os  
from PIL import Image  

# 自定义数据集类  
class CustomImageDataset(Dataset):  
    def __init__(self, images_dir, labels_csv, transform=None):  
        self.images_dir = images_dir  
        self.transform = transform  
        self.labels_df = pd.read_csv(labels_csv)  
        self.label_map = {row['img_name']: row['label'] for _, row in self.labels_df.iterrows()}  
        self.image_files = list(self.label_map.keys())  

    def __len__(self):  
        return len(self.image_files)  

    def __getitem__(self, idx):  
        img_name = self.image_files[idx]  
        img_path = os.path.join(self.images_dir, img_name)  
        image = Image.open(img_path)  
        label = self.label_map[img_name]  

        if self.transform:  
            image = self.transform(image)  

        return image, label  

# 测试数据集类  
class TestDataset(Dataset):  
    def __init__(self, images_dir, transform=None):  
        self.images_dir = images_dir  
        self.transform = transform  
        self.image_files = os.listdir(images_dir)  

    def __len__(self):  
        return len(self.image_files)  

    def __getitem__(self, idx):  
        img_name = self.image_files[idx]  
        img_path = os.path.join(self.images_dir, img_name)  
        image = Image.open(img_path)  

        if self.transform:  
            image = self.transform(image)  

        return image, img_name  

# 数据预处理  
transform = transforms.Compose([  
    transforms.Resize(256),  
    transforms.CenterCrop(224),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])  

# 加载数据集  
train_image_dir = '/tcdata/train/img'  
train_labels_csv = '/tcdata/train/label.csv'  
test_image_dir = '/tcdata/test/img'  

train_dataset = CustomImageDataset(train_image_dir, train_labels_csv, transform=transform)  
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

# 加载预训练的 DenseNet121 模型  
model = timm.create_model('densenet121', pretrained=True)  
num_features = model.classifier.in_features  
model.classifier = nn.Linear(num_features, 2)  # 二分类  
model = model.cuda()  

# 定义损失函数和优化器  
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)  

# 训练模型  
num_epochs = 10  
best_loss = float('inf')  
best_model_wts = None  

for epoch in range(num_epochs):  
    model.train()  
    running_loss = 0.0  
    for inputs, labels in train_loader:  
        inputs, labels = inputs.cuda(), labels.cuda()  
        
        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  
        
        running_loss += loss.item() * inputs.size(0)  
    
    epoch_loss = running_loss / len(train_loader.dataset)  
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')  
    
    if epoch_loss < best_loss:  
        best_loss = epoch_loss  
        best_model_wts = model.state_dict()  

model.load_state_dict(best_model_wts)  

# 测试模型并输出结果到CSV  
test_dataset = TestDataset(test_image_dir, transform=transform)  
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  

model.eval()  
results = []  

with torch.no_grad():  
    for inputs, image_names in test_loader:  
        inputs = inputs.cuda()  
        outputs = model(inputs)  
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 获取类别1的概率  
        for img_name, prob in zip(image_names, probabilities.cpu().numpy()):  
            results.append({'case': img_name, 'prob': prob})  

# 保存结果到CSV  
results_df = pd.DataFrame(results)  
results_df.to_csv('classification_results.csv', index=False)