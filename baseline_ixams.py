from config import *
from data_loader_ixmas import load_ixmas_data, select_domain_data
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import time
import numpy as np

if __name__ == '__main__':
    # 1. 数据准备
    params = config  # 使用IXMAS配置
    
    # 加载所有数据
    full_loader, full_stats = load_ixmas_data()
    print("完整数据集统计:", full_stats)
    
    # 获取numpy数组格式数据
    all_features = []
    all_categories = []
    all_domains = []
    for batch in full_loader:
        all_features.append(batch[0].numpy())
        all_categories.append(batch[1].numpy())
        all_domains.append(batch[2].numpy())
    
    # 合并数据
    X_all = np.concatenate(all_features)
    y_all = np.concatenate(all_categories)
    domains = np.concatenate(all_domains)
    
    # 2. 划分源域和目标域
    # 源域: 摄像机0-3
    src_idx = np.isin(domains, [2,3,4])
    X_src = X_all[src_idx]
    y_src = y_all[src_idx]
    
    # 目标域: 摄像机4
    tgt_idx = np.isin(domains, [0,1])
    X_tgt = X_all[tgt_idx]
    y_tgt = y_all[tgt_idx]
    
    # 3. 训练标准SVM模型
    print("训练标准SVM模型...")
    svm_model = SVC(
        kernel='linear',  # 使用线性核
        C=1.0,            # 正则化参数
        decision_function_shape='ovr'  # 一对多策略
    )
    
    start = time.time()
    svm_model.fit(X_src, y_src)
    end = time.time()
    print(f'训练完成，耗时 {end-start:.2f}秒')
    
    # 4. 在源域和目标域上评估
    # 源域评估
    y_src_pred = svm_model.predict(X_src)
    src_accuracy = accuracy_score(y_src, y_src_pred)
    print(f"源域准确率: {src_accuracy:.4f}")
    print("源域混淆矩阵:\n", confusion_matrix(y_src, y_src_pred))
    
    # 目标域评估
    y_tgt_pred = svm_model.predict(X_tgt)
    if len(y_tgt) > 0:
        tgt_accuracy = accuracy_score(y_tgt, y_tgt_pred)
        print(f"目标域准确率: {tgt_accuracy:.4f}")
        print("目标域混淆矩阵:\n", confusion_matrix(y_tgt, y_tgt_pred))
    else:
        print("目标域无样本，无法评估")
    
    # 5. 保存模型信息
    model_info = {
        'model': svm_model,
        'training_time': end - start,
        'source_accuracy': src_accuracy,
        'target_accuracy': tgt_accuracy if len(y_tgt) > 0 else None,
        'feature_dim': X_src.shape[1]
    }
    
    print("模型训练完成:", model_info)