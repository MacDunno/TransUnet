import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from PIL import Image
import os


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


# def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
#     image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
#     x, y = image.shape[0], image.shape[1]
#     if x != patch_size[0] or y != patch_size[1]:
#         #缩放图像符合网络输入
#         image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
#     input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
#     net.eval()
#     with torch.no_grad():
#         out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
#         out = out.cpu().detach().numpy()
#         if x != patch_size[0] or y != patch_size[1]:
#             #缩放图像至原始大小
#             prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
#         else:
#             prediction = out
    
#     metric_list = []
#     for i in range(1, classes):
#         metric_list.append(calculate_metric_percase(prediction == i, label == i))

#     if test_save_path is not None:
#         img_itk = sitk.GetImageFromArray(image.astype(np.float32))
#         prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
#         lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
#         img_itk.SetSpacing((1, 1, z_spacing))
#         prd_itk.SetSpacing((1, 1, z_spacing))
#         lab_itk.SetSpacing((1, 1, z_spacing))
#         sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
#         sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
#         sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
#     return metric_list

def save_as_png(array, save_path, file_name_prefix):  
    """  
    将 3D 图像的每一层切片保存为 PNG 格式。  
    :param array: 3D 图像的 numpy 数组，形状为 (Depth, Height, Width)  
    :param save_path: 保存路径  
    :param file_name_prefix: 保存文件的前缀  
    """  
    os.makedirs(save_path, exist_ok=True)  # 确保保存路径存在  
    depth = array.shape[0]  # 获取切片数（即深度）  

    for i in range(depth):  
        slice_2d = array[i, :, :]  # 提取第 i 层  
        slice_norm = ((slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min()) * 255).astype(np.uint8)  # 归一化到 [0, 255]  
        save_file = os.path.join(save_path, f"{file_name_prefix}_slice_{i}.png")  
        Image.fromarray(slice_norm).save(save_file)  # 使用 Pillow 保存为 PNG 格式  

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):  
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()  
    x, y = image.shape[0], image.shape[1]  
    if x != patch_size[0] or y != patch_size[1]:  
        # 缩放图像符合网络输入  
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)  
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()  
    net.eval()  
    with torch.no_grad():  
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)  
        out = out.cpu().detach().numpy()  
        if x != patch_size[0] or y != patch_size[1]:  
            # 缩放图像至原始大小  
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)  
        else:  
            prediction = out  
    
    metric_list = []  
    for i in range(1, classes):  
        metric_list.append(calculate_metric_percase(prediction == i, label == i))  

    if test_save_path is not None:  
        # 以 PNG 格式保存原始图像、预测结果以及标签  
        # save_as_png(image.astype(np.float32), os.path.join(test_save_path, case + "_img"), case + "_img")  
        save_as_png(prediction.astype(np.float32), os.path.join(test_save_path, case + "_pred"), case + "_pred")  
        # save_as_png(label.astype(np.float32), os.path.join(test_save_path, case + "_gt"), case + "_gt")  

    return metric_list  

def test_only(image, net, classes, test_save_path, patch_size=[256, 256], case=None, z_spacing=1):
    image = image.squeeze(0).cpu().detach().numpy()
    x, y = image.shape[0], image.shape[1]
    if x != patch_size[0] or y != patch_size[1]:
        #缩放图像符合网络输入
        image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            #缩放图像至原始大小
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out
    prediction_pil = Image.fromarray((prediction * 2/classes).astype(np.uint8))
    prediction_pil.save(f"{test_save_path}/{case}.png") 
    return