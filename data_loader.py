import os
import numpy as np
from PIL import Image
from datetime import datetime

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import scipy.io as sio


# 配置参数
class Config:
    img_dir = "./office_caltech_10/" 
    # save_path = "./office_caltech_dl_ms0.mat"  
    batch_size = 32 
    input_size = 227
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "alexNet" 

    DOMAINS = ['amazon', 'caltech', 'dslr', 'webcam']
    CATEGORIES = ['back_pack', 'bike', 'calculator', 'headphones',
                  'keyboard', 'laptop_computer', 'monitor', 'mouse', 'mug', 'projector']

class DomainDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_paths = []
        self.domain_labels = []
        self.category_labels = []

        for domain in os.listdir(img_dir):
            if domain.lower() not in Config.DOMAINS:
                raise ValueError(f"检测到非法领域目录: {domain}，合法领域应为{Config.DOMAINS}")

            domain_path = os.path.join(img_dir, domain)
            if os.path.isdir(domain_path):
                for category in os.listdir(domain_path):
                    if category not in Config.CATEGORIES:
                        raise ValueError(f"检测到非法类别目录: {category}，合法类别应为{Config.CATEGORIES}")

                    category_path = os.path.join(domain_path, category)
                    for img_file in os.listdir(category_path):
                        if img_file.endswith(('.jpg', '.png')):
                            self.img_paths.append(os.path.join(category_path, img_file))
                            self.domain_labels.append(Config.DOMAINS.index(domain.lower()))
                            self.category_labels.append(Config.CATEGORIES.index(category))

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        return self.transform(img), self.category_labels[idx], self.domain_labels[idx]

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        original = models.alexnet(pretrained=True)

        # 结构切片策略
        self.features = original.features  # Conv layers
        self.avgpool = original.avgpool  # AdaptiveAvgPool2d
        self.classifier = torch.nn.Sequential(*list(
            original.classifier.children())[:-1]  # 保留前两个FC层
                                              )

    def forward(self, x):
        x = self.features(x)  # 输出形状: [N, 256, 6, 6]
        x = self.avgpool(x)  # 保持形状: [N, 256, 6, 6]
        x = torch.flatten(x, 1)  # 展平为[N, 256*6*6=9216]
        x = self.classifier(x)  # 输出[N, 4096]
        return x

# 增强版主处理流程
def preprocessing():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(Config.input_size),  # 调整剪裁尺寸
        transforms.ToTensor(),
        # AlexNet专用归一化参数（与ResNet不同）
        transforms.Normalize(mean=[0.485, 0.406, 0.456],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = DomainDataset(Config.img_dir, transform)
    loader = DataLoader(dataset, batch_size=Config.batch_size,
                        shuffle=False, num_workers=4)

    # __model = FeatureExtractor(Config.model_name).to(Config.device)
    model = FeatureExtractor().to(Config.device)
    model.eval()

    features = torch.zeros(len(dataset), 4096)
    domain_labels = np.zeros(len(dataset), dtype=np.int16)
    category_labels = np.zeros(len(dataset), dtype=np.int16)

    with torch.no_grad():
        for i, (imgs, categories, domains) in enumerate(loader):
            imgs = imgs.to(Config.device)
            batch_features = model(imgs)

            start = i * Config.batch_size
            end = start + imgs.size(0)
            features[start:end] = batch_features.cpu()
            category_labels[start:end] = categories.numpy()
            domain_labels[start:end] = domains.numpy()

            print(f"进度: {i + 1}/{len(loader)} | 当前批次样本: {imgs.size(0)}")

    # 构建适配select_domain_data的字典结构
    output_dict = {
        'features': features.numpy().astype(np.float32),  # 保持(N,4096)形状
        'domain_labels': domain_labels.astype(np.int32),
        'category_labels': category_labels.astype(np.int32),
        'feature_dim': 4096,
        'sample_count': len(dataset),
        '_raw_domains': np.array(Config.DOMAINS)[domain_labels],  # 可选调试信息
        '_raw_categories': np.array(Config.CATEGORIES)[category_labels]
    }
    return output_dict


import numpy as np
from typing import Dict, List


def select_domain_data(data: Dict, domain_ids: List[int]) -> Dict:
    # 生成域选择掩码
    domain_mask = np.isin(data['domain_labels'], domain_ids)

    # 空域处理
    if np.sum(domain_mask) == 0:
        print(f"[警告] 未找到目标域: {domain_ids}")
        return {}

    # 提取子集数据
    selected_features = data['features'][domain_mask]
    selected_categories = data['category_labels'][domain_mask]

    # 数据验证
    unique_categories = np.unique(selected_categories)
    category_count = len(unique_categories)
    assert selected_categories.max() == category_count - 1, "类别标签应从0开始连续编号"

    sample_count, feature_dim = selected_features.shape
    assert len(selected_categories) == sample_count, "样本数量不一致"
    assert feature_dim == data['feature_dim'], f"特征维度应为{data['feature_dim']}"

    # 输出统计信息
    print(f"选中域: {domain_ids} | 类别数: {category_count} | 样本量: {sample_count} | 维度: {feature_dim}")

    return {
        'features': selected_features.astype(np.float32),
        'category_labels': selected_categories.astype(np.int32),
        'selection_mask': domain_mask,
        'domain_labels': data['domain_labels'][domain_mask],
        'category_count': category_count,
        'feature_dim': feature_dim,
        'sample_count': sample_count
    }
