import os
import numpy as np
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader

class IXMASConfig:
    # 配置参数
    feature_root = "/home/r22user1/linziyue/7404Aproject/ixmas/ixmas_reorganized"  # 特征文件根目录
    domains = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4']  # 摄像头视角
    categories = [  # 动作类别列表
        'check-watch', 'cross-arms',  
        'sit-down', 'get-up', 'scratch-head']
      # 请根据您的实际动作类别修改
    feature_dim = 5000  # 根据您的特征维度修改

class IXMASDataset(Dataset):
    def __init__(self, feature_root, domains=None, categories=None):
        """
        初始化数据集（适配 cam/action 目录结构）
        :param feature_root: 特征文件根目录
        :param domains: 指定要加载的domain列表(cam0-4)，None表示加载全部
        :param categories: 指定要加载的category列表(动作)，None表示加载全部
        """
        self.feature_paths = []
        self.domain_labels = []
        self.category_labels = []
        
        # 使用配置中的默认值
        if domains is None:
            domains = IXMASConfig.domains
        if categories is None:
            categories = IXMASConfig.categories
            
        # 新的目录结构: feature_root/cam/action/xxx_bow.npy
        for cam in os.listdir(feature_root):
            if cam not in domains:
                continue
                
            cam_path = os.path.join(feature_root, cam)
            if not os.path.isdir(cam_path):
                continue
                
            for action in os.listdir(cam_path):
                if action not in categories:
                    continue
                    
                action_path = os.path.join(cam_path, action)
                if not os.path.isdir(action_path):
                    continue
                    
                # 加载所有_bow.npy文件
                for file in os.listdir(action_path):
                    if file.endswith('_bow.npy'):
                        file_path = os.path.join(action_path, file)
                        self.feature_paths.append(file_path)
                        self.domain_labels.append(IXMASConfig.domains.index(cam))
                        self.category_labels.append(IXMASConfig.categories.index(action))
        
        # 验证数据
        if len(self.feature_paths) == 0:
            raise ValueError(f"未找到任何特征文件，请检查路径: {feature_root} 和过滤条件 domains={domains}, categories={categories}")

    def __len__(self):
        return len(self.feature_paths)
    
    def __getitem__(self, idx):
        # 加载npy文件
        feature = np.load(self.feature_paths[idx])
        # 转换为tensor并确保是float32类型
        feature = torch.from_numpy(feature).float()
        # 返回特征、动作类别、摄像头视角
        return feature, self.category_labels[idx], self.domain_labels[idx]

def load_ixmas_data(domains=None, categories=None, batch_size=32):
    """
    加载IXMAS数据集
    :param domains: 要加载的摄像头视角列表，None表示全部
    :param categories: 要加载的动作类别列表，None表示全部
    :param batch_size: 批大小
    :return: DataLoader和数据统计信息
    """
    # 创建数据集
    dataset = IXMASDataset(IXMASConfig.feature_root, domains, categories)
    
    # 创建数据加载器
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 数据统计信息
    stats = {
        'total_samples': len(dataset),
        'domains_loaded': [IXMASConfig.domains[i] for i in np.unique(dataset.domain_labels)],
        'categories_loaded': [IXMASConfig.categories[i] for i in np.unique(dataset.category_labels)],
        'feature_dim': IXMASConfig.feature_dim
    }
    
    return loader, stats

def select_domain_data(features, domain_labels, category_labels, domain_ids):
    """
    选择特定domain的数据
    :param features: 所有特征数据 (numpy array)
    :param domain_labels: 所有domain标签 (numpy array)
    :param category_labels: 所有category标签 (numpy array)
    :param domain_ids: 要选择的domain ID列表
    :return: 筛选后的特征、category标签和统计信息
    """
    # 生成域选择掩码
    domain_mask = np.isin(domain_labels, domain_ids)
    
    # 空域处理
    if np.sum(domain_mask) == 0:
        print(f"[警告] 未找到目标域: {domain_ids}")
        return None, None, None
    
    # 提取子集数据
    selected_features = features[domain_mask]
    selected_categories = category_labels[domain_mask]
    selected_domains = domain_labels[domain_mask]
    
    # 数据统计
    stats = {
        'selected_domains': [IXMASConfig.domains[i] for i in np.unique(selected_domains)],
        'category_distribution': {IXMASConfig.categories[i]: count 
                                for i, count in zip(*np.unique(selected_categories, return_counts=True))},
        'sample_count': len(selected_features),
        'feature_dim': selected_features.shape[1]
    }
    
    return selected_features, selected_categories, stats

# 使用示例
if __name__ == "__main__":
    # 示例1: 加载所有数据
    loader, stats = load_ixmas_data()
    print("加载所有数据统计:", stats)
    
    # 示例2: 只加载cam0和cam1的数据
    loader, stats = load_ixmas_data(domains=['cam0', 'cam1'])
    print("\n加载cam0和cam1数据统计:", stats)
    
    # 示例3: 只加载特定动作的数据
    loader, stats = load_ixmas_data(categories=['check-watch', 'walk'])
    print("\n加载特定动作数据统计:", stats)
    
    # 示例4: 从已加载数据中选择特定domain
    # 假设我们已经有了所有特征数据
    all_features = np.concatenate([batch[0].numpy() for batch in loader])
    all_domains = np.concatenate([batch[2].numpy() for batch in loader])
    all_categories = np.concatenate([batch[1].numpy() for batch in loader])
    
    selected_feats, selected_cats, stats = select_domain_data(
        all_features, all_domains, all_categories, [0, 1]  # 选择cam0和cam1
    )
    print("\n从已加载数据中选择cam0和cam1:", stats)