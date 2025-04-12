import numpy as np
import time

import torch
from scipy.optimize import minimize
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
from data_loader import *

class LRELSSVM:
    def __init__(self, C1=10, C2=1, lambda1=10, lambda2=10, K=5):
        self.C1 = C1 
        self.C2 = C2 
        self.lambda1 = lambda1  
        self.lambda2 = lambda2  
        self.K = K  
        self.W = []  # 权重矩阵c个，期望形状：[d x n]（d为特征数，n为正样本总数）
        self.F = None  # 辅助低秩矩阵，形状：[n x n]
        self.M_inv = None


    def _update_F(self, G):
        """使用SVT更新F矩阵，并打印SVT信息"""
        U, S, Vt = torch.linalg.svd(G, full_matrices=False)
        threshold = self.lambda1 / (2 * self.lambda2)
        print(f"SVT update: 原始S值: {S[:5].cpu().numpy()}..., 阈值: {threshold}")
        S_thresh = torch.maximum(S - threshold, torch.tensor(0))
        diff_F = torch.linalg.norm(U @ torch.diag(S_thresh) @ Vt - G, ord='fro')
        print(f"SVT update: trace(S_thresh) = {torch.sum(S_thresh):.4f}, ||F - G||_F = {diff_F:.6f}")
        return U @ torch.diag(S_thresh) @ Vt

    def _solve_exemplar_lssvm(self, x_pos, X_pos, X_neg, f_col, m_neg, M_inv):
        """快速求解单个exemplar LSSVM问题（公式12）"""
        d = 4096 
        # d = 4  
        if X_pos.shape[0] != d:
            X_pos = X_pos.T  # 变为 [d x n_pos]
        if x_pos.shape[0] != d:
            x_pos = x_pos.T  # 变为 [d x n_pos]
        if X_neg.shape[0] != d:
            X_neg = X_neg.T  # 变为 [d x n_neg]

       
        X_ext = torch.hstack([x_pos, X_neg, X_pos]).to(Config.device)
        # 计算 M = X_ext^T X_ext + D
        m1 = (X_ext.T @ x_pos).to(Config.device)
        m11 = m1[0][0] + 1 / self.C1
        m1 = m1[1:]
        miu = 1/(m11-m1.T @ M_inv @ m1)
        M_t_inv = torch.hstack([torch.vstack([miu, -miu*(M_inv @ m1)]),
                                torch.vstack([-miu*(M_inv @ m1).T,
                                              M_inv+miu* M_inv @ m1 @ m1.T @ M_inv])])
        # 构造右侧向量 y = [1, -1 (m_neg times), f_col]
        y_vec = torch.cat([torch.tensor([1]).to(Config.device),
                           torch.full((m_neg,), -1).to(Config.device),
                           f_col])
        # 求解 alpha = M_inv @ y_vec
        alpha = M_t_inv @ y_vec
        w = X_ext @ alpha
        return w

    def fit(self, X_train, y_train, max_iter=10):
        """
        训练 LRE-LSSVM 模型。
        """
        n_samples, d = X_train.shape
        lb = LabelBinarizer()
        Y_bin = lb.fit_transform(y_train)  # [n_samples x n_classes]
        cate_num = Y_bin.shape[1]
        print("训练数据统计:")
        print("  特征形状:", X_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("  类别标签分布:", (unique, counts))

        X_pos_list = [torch.tensor(X_train[y_train == c].T) for c in range(cate_num)]
        X_neg_list = [torch.tensor(X_train[y_train != c].T) for c in range(cate_num)]
        # 统计各类别正样本数
        pos_counts = [x.shape[1] for x in X_pos_list]
        print("各类别正样本数量:", pos_counts)

        n_total = sum(pos_counts)
        self.W = [None for _ in range(cate_num)]
        threshold_F, threshold_W = 1e-3, 1e-4
        for c in range(cate_num):
            X_pos = X_pos_list[c].to(Config.device)
            X_neg = X_neg_list[c].to(Config.device)
            n_c = X_pos.shape[1]
            W = torch.randn(d, n_c).to(Config.device) * 0.01
            n_pos = X_pos.shape[1]
            m_neg = X_neg.shape[1]
            D_ = torch.diag(torch.cat([
                torch.full((m_neg,), 1 / self.C2),
                torch.full((n_pos,), 1 / self.lambda2)
            ])).to(Config.device)
            M_X_ext = torch.hstack([X_neg, X_pos])
            M = M_X_ext.T @  M_X_ext + D_
            M_inv = torch.linalg.inv(M)
            pre_F = F = torch.zeros((n_c, n_c)).to(Config.device)
            pre_W = torch.zeros((d, n_c)).to(Config.device)
            for it in range(max_iter):

                W_cols = []
                for j in range(n_c):
                    print(end=f'\rcalculate exemplar: {j+1}/{n_c}, for category: {c}, iter: {it+1}/{max_iter}')
                    f_col = F[:, j]
                    x_pos = X_pos[:, j:j + 1]
                    w_exemplar = self._solve_exemplar_lssvm(x_pos, X_pos, X_neg, f_col, m_neg, M_inv)
                    W_cols.append(w_exemplar.reshape(-1, 1))
                print()
                W = torch.hstack(W_cols)
                G = W.T @ X_pos
                F = self._update_F(G)
                
                W_change = torch.mean(torch.abs(W - pre_W))
                F_change = torch.mean(torch.abs(F - pre_F))

                print(f"迭代 {it + 1} 完成, W: mean = {torch.mean(W).item():.4e}, std = {torch.std(W).item():.4e}, "
                      f"W change: {W_change:.4e}, F cahnge:{F_change:.4e}\n"
                      f"--------------------------------------------------------------")
                if W_change < threshold_W:
                    break
                pre_F, pre_W = F, W
            self.W[c] = W.cpu().numpy()
        return self

    def predict(self, X_test):
        """领域泛化预测（类似公式20），采用 top-K 融合策略"""
        n_c = len(self.W)
        n_test = X_test.shape[0]
        pred_vals = np.zeros((n_test, n_c))
        for c in range(n_c):
            scores = X_test @ self.W[c]  # 形状 [n_test x n_total]
            prob = 1.0 / (1.0 + np.exp(-scores))
            for i in range(n_test):
                # 对每个测试样本，选择 top-K 得分
                sample_scores = prob[i, :]
                topk = np.partition(sample_scores, -self.K)[-self.K:]
                pred_vals[i][c] = np.mean(topk)
        return pred_vals


def evaluate_model(model, train_data, test_data, max_iter=10):
    start_time = time.time()
    model.fit(train_data['features'], train_data['category_labels'], max_iter=max_iter)
    print(f"Training time: {time.time() - start_time:.2f}s")

    test_preds = model.predict(test_data['features'])
    test_labels = test_data['category_labels']

    pred_label = [np.argmax(i) for i in test_preds]

    acc = accuracy_score(test_labels, pred_label)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc


def save_model(model, param):
    model_path = param.get('model_path', './models/')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    save_file = os.path.join(model_path,
                             f"lre_lssvm_model_C{param['C1']}_W{param['C2']}_L{param['lambda1']}-{param['lambda2']}.pkl")
    with open(save_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"模型已保存到: {save_file}")

def l2_normalize_features(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / (norms + 1e-8)

if __name__ == "__main__":
    mat_file_path = 'data/office_caltech_dl_ms0.mat'
    data_dict = load_mat_data(mat_file_path)
    py_data = convert_mat_to_py_dict(data_dict)

    # 打印转换后字典的关键信息
    print("转换后的数据字典：")
    print(f"  特征矩阵形状: {py_data['features'].shape}")
    print(f"  域标签形状: {py_data['domain_labels'].shape}, 示例: {py_data['domain_labels'][:10]}")
    print(f"  类别标签形状: {py_data['category_labels'].shape}, 示例: {py_data['category_labels'][:10]}")
    print(f"  样本数量: {py_data['sample_count']}")
    print(f"  特征维度: {py_data['feature_dim']}")
    print(f"  原始域名称示例: {py_data['_raw_domains'][:10]}")
    print(f"  原始类别名称示例: {py_data['_raw_categories'][:10]}")

    features_norm = l2_normalize_features(py_data['features'])
    py_data['features'] = features_norm

    train_selected = select_domain_data(py_data, [0, 1])
    test_selected = select_domain_data(py_data, [2, 3])
    train_data = {
        'features': train_selected['features'],
        'category_labels': train_selected['category_labels']  # 转换为 1-indexed
    }

    test_data = {
        'features': test_selected['features'],
        'category_labels': test_selected['category_labels']
    }

    # 打印训练数据统计信息
    print("训练数据统计:")
    print("  特征形状:", train_data['features'].shape)
    print("  类别标签分布:", np.unique(train_data['category_labels'], return_counts=True))

    model_instance = LRELSSVM(C1=10, C2=1, lambda1=10, lambda2=10, K=5)
    evaluate_model(model_instance, train_data, test_data, max_iter=100)
    save_model(model_instance, {'model_path': './models/', 'C1': 10, 'C2': 1, 'lambda1': 10, 'lambda2': 10})
