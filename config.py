import os
from dataclasses import dataclass

config = {
    # 基础参数
    'dam_flag': 0,  # 领域模式 0:泛化 1:适应
    'save_model_flag': 2,  # 模型保存模式 0/1/2
    'model_path': r'./model',  # 模型存储路径

    # 优化参数
    'svm_C': 0.001,  # SVM正则化系数
    'lambda1': 1.0,  # 主损失权重
    'lambda2': 4.0,  # 正则项权重
    'exemplar_weight': 10.0,  # 样本权重
    'prdct_top_num': 5,  # Top-K预测数

    # 域适应参数
    'mmd_sig': 1.0,  # MMD核宽
    'dam_sig': 1.0,  # 适应核参数
    'dam_g1': 1.0,  # 权重γ1
    'dam_g2': 1.0,  # 权重γ2
    'dam_eps': 1e-3,  # 收敛阈值
    'dam_min_stop': 1e-5,  # 停止条件

    # 训练控制
    'max_ite': 10,  # 外循环次数
    'max_inner_ite': 100,  # 内循环步数
    'min_f_thred': 1e-7,
    'min_obj_val': 1e-7,  # 损失阈值
    # 'dam_min_stop': 1e-5  # 变化阈值
}
