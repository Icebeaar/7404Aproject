from config import *
from data_loader_ixmas import load_ixmas_data, select_domain_data
from LRE_SVMs_train import LRE_SVM_Trainer
from LRE_SVMs_predict import LRE_SVMs_Predictor
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import numpy as np

if __name__ == '__main__':
    # 1. 数据准备
    params = config  # 使用IXMAS配置
    
    # 加载所有数据（自动检测特征维度）
    full_loader, full_stats = load_ixmas_data()
    print("完整数据集统计:", full_stats)
    
    # 获取numpy数组格式数据并自动确定特征维度
    all_features = []
    all_categories = []
    all_domains = []
    for batch in full_loader:
        all_features.append(batch[0].numpy())
        all_categories.append(batch[1].numpy())
        all_domains.append(batch[2].numpy())
    
    # 自动从第一个样本获取特征维度
    feature_dim = all_features[0].shape[1] if len(all_features[0].shape) > 1 else 1
    
    datadict = {
        'features': np.concatenate(all_features),
        'category_labels': np.concatenate(all_categories),
        'domain_labels': np.concatenate(all_domains),
        'feature_dim': feature_dim  # 自动确定的特征维度
    }
    
    # 2. 划分源域和目标域
    src_data = select_domain_data(
        datadict['features'], 
        datadict['domain_labels'], 
        datadict['category_labels'], 
        domain_ids=[0,1,2,3]  # cam0和cam1作为源域
    )
    
    tgt_data = select_domain_data(
        datadict['features'], 
        datadict['domain_labels'], 
        datadict['category_labels'], 
        domain_ids=[4]  # cam2-4作为目标域
    )
    
    # 3. 训练和评估
    model = LRE_SVM_Trainer(params)
    
    start = time.time()
    model_dict = model.train({
        'features': src_data[0],  # 源域特征
        'category_labels': src_data[1],  # 源域标签
        'feature_dim': feature_dim  # 传递实际特征维度
    })
    end = time.time()
    print(f'Training process took {end-start:.2f}s ----------')
    
    # 在目标域测试
    predictor = LRE_SVMs_Predictor(model_dict)
    predictions = predictor.predict({
        'features': tgt_data[0],  # 目标域特征
        'category_labels': tgt_data[1]  # 目标域标签（用于评估）
    })
    
    # 计算指标
    if tgt_data[1] is not None:
        y_true = tgt_data[1]
        y_pred = predictions
        print("目标域准确率:", accuracy_score(y_true, y_pred))
        print("混淆矩阵:\n", confusion_matrix(y_true, y_pred))
    
    print("训练结果模型字典:", model_dict)