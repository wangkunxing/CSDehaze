import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def is_image_file(filename):
    """
    判断文件是否为图像文件。
    """
    return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp'])

class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=256):
        assert mode in ['train', 'valid', 'test'], "Mode must be one of 'train', 'valid', 'test'"
        
        self.mode = mode  # 数据模式
        self.size = size  # 输出图像尺寸
        
        self.root_dir = os.path.join(data_dir, sub_dir)  # 数据子目录路径
        
        # 获取所有合法的图像文件名列表，过滤掉非图像文件和隐藏文件
        gt_path = os.path.join(self.root_dir, 'GT')
        self.img_names = sorted([f for f in os.listdir(gt_path) if is_image_file(f)])
        self.img_num = len(self.img_names)  # 图像数量

        # 根据模式选择预处理操作
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((3840, 3840), interpolation=transforms.InterpolationMode.BILINEAR),  # 先将图像缩放到 1024
                transforms.RandomHorizontalFlip(),  # 水平翻转
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif mode == 'valid':
            self.transform = transforms.Compose([
                transforms.Resize((3840, 3840), interpolation=transforms.InterpolationMode.BILINEAR),  # 先将图像缩放到 1024
                transforms.ToTensor(),  # 转换为 Tensor 并自动归一化到 [0, 1]
                transforms.CenterCrop(2048),  # 中心裁剪
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
            ])
        else:  # test
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        """返回数据集中图像的数量。"""
        return self.img_num

    def __getitem__(self, idx):
        """
        根据索引获取数据项。

        参数:
            idx (int): 数据索引。

        返回:
            dict: 包含源图像、目标图像和文件名的数据字典。
        """
        img_name = self.img_names[idx]

        source_img = Image.open(os.path.join(self.root_dir, 'hazy', img_name)).convert('RGB')
        target_img = Image.open(os.path.join(self.root_dir, 'GT', img_name)).convert('RGB')

        source_img = self.transform(source_img)
        target_img = self.transform(target_img)

        return {'source': source_img, 'target': target_img, 'filename': img_name}