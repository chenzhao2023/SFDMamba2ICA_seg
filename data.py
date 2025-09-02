import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as TF

# 定义灰度图的 transforms
transform = transforms.Compose([
    transforms.ToTensor()  # 将图像数据转换为张量格式（PyTorch张量），并且自动将像素值缩放到 [0, 1] 之间
])

def resize_image(path, size=(512, 512)):
    """加载并调整图像大小"""
    img = Image.open(path).convert('L')  # 确保加载为灰度图
    img = img.resize(size)
    return img

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'ICA_PNG'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        segment_path = os.path.join(self.path, 'label', segment_name)
        image_path = os.path.join(self.path, 'ICA_PNG', segment_name)

        # 加载并调整大小为灰度图
        segment_image = resize_image(segment_path)
        image = resize_image(image_path)

        # 返回转换后的张量
        return transform(image), transform(segment_image), segment_name

if __name__ == '__main__':
    # 测试数据集
    data = MyDataset(r"dataset/test")
    print("输入图像形状:", data[0][0].shape)  # (1, H, W)
    print("目标图像形状:", data[0][1].shape)  # (1, H, W)
