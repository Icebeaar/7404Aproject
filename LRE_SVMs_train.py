from gc import set_threshold

import numpy as np
from scipy.optimize import minimize
import pickle
import os

import numpy as np
import pickle
import os
from scipy.optimize import minimize
from scipy.linalg import svd
from time import time
from sklearn.metrics import accuracy_score, confusion_matrix

class LRE_SVM_Trainer:
    def __init__(self, param):
        self.param = param
        self.model = {
            'esvm': {'esvm_weights': None, 'esvm_bias': None, 'train_lbl': None},
            'cate_num': 0,
            'dam_flag': param.get('dam_flag', 0),
            'prdct_top_num': param.get('prdct_top_num', 5),
            'dam_C': param.get('dam_g1', 1.0),
            'dam_lambda': param.get('dam_g2', 0.1),
            'dam_eps': param.get('dam_eps', 1e-6)
        }

    def evaluate_training_accuracy(self, model, train_ftr, train_lbl):
        """
        根据当前模型对训练数据进行预测，并计算训练准确率和混淆矩阵。
        同时，对于每个类别，打印该类别在训练样本上的 topK 预测分数的均值和标准差。
        参数：
          model: 训练得到的模型字典，至少应包含键：
                 - 'esvm'：字典，其中包含 'esvm_weights' (shape: [num_exemplars, ftr_dim]),
                              'esvm_bias' (shape: [num_exemplars, 1]) 和 'train_lbl' (1-indexed)。
          train_ftr: 训练特征矩阵，形状为 (num_samples, ftr_dim)
          train_lbl: 训练标签（1-indexed），形状为 (num_samples,)
        返回：
          acc: 训练准确率（0~1）
          conf_mat: 混淆矩阵
        """
        esvm_weights = model['esvm']['esvm_weights']  # shape: (num_exemplars, ftr_dim)
        esvm_bias = model['esvm']['esvm_bias']  # shape: (num_exemplars, 1)
        train_labels_model = model['esvm']['train_lbl']  # shape: (num_samples,), 1-indexed

        # 计算每个 exemplar 对所有训练样本的预测值
        # 注意：train_ftr.T 的 shape 为 (ftr_dim, num_samples)
        esvm_predict_val = esvm_weights @ train_ftr.T + esvm_bias
        # 采用 logistic 函数转换为概率
        esvm_predict_prob = 1.0 / (1.0 + np.exp(-esvm_predict_val))

        # 如果模型采用 TopK 融合策略
        prdct_top_num = model.get('prdct_top_num', 5)
        num_samples = train_ftr.shape[0]
        cate_num = model['cate_num']
        predict_val = np.zeros((num_samples, cate_num))

        # 对每个类别，找到对应 exemplar 的索引（假设模型中存储的 train_lbl 即为 exemplar 对应的类别）
        for cate in range(1, cate_num + 1):
            cate_idx = np.where(train_labels_model == cate)[0]
            # 对每个训练样本，将该类别下所有 exemplar 的预测分数排序后取前 K 个求和
            for i in range(num_samples):
                scores = esvm_predict_prob[cate_idx, i]
                sorted_scores = np.sort(scores)[::-1]  # 降序排序
                topk_sum = np.sum(sorted_scores[:prdct_top_num])
                predict_val[i, cate - 1] = topk_sum

        # 得到预测标签：取各类别分数最高的类别（记得加回 1 以保证 1-indexed）
        pred_lbl = np.argmax(predict_val, axis=1) + 1
        acc = accuracy_score(train_lbl, pred_lbl)
        conf_mat = confusion_matrix(train_lbl, pred_lbl)

        print("训练准确率: {:.2f}%".format(acc * 100))
        print("训练混淆矩阵:")
        print(conf_mat)

        # 对每个类别打印预测分数的均值和标准差
        for cate in range(1, cate_num + 1):
            scores_cate = predict_val[:, cate - 1]
            print(f"类别 {cate}: topK 分数均值 = {np.mean(scores_cate):.4f}, 标准差 = {np.std(scores_cate):.4f}")

        return acc, conf_mat
    def train(self, data):
        train_ftr = data['features']
        train_lbl = data['category_labels'].flatten()
        # print(train_ftr.shape)
        trn_smpl_num, ftr_dim = train_ftr.shape

        cate_num = np.max(train_lbl)

        weights = np.zeros_like(train_ftr)
        bias = np.zeros(trn_smpl_num)

        for cate_i in range(1, cate_num+1):
            print(f"Training category {cate_i}/{cate_num}")
            cate_idx = (train_lbl == cate_i)
            tmp_trn_lbl = 2*cate_idx.astype(int) - 1  # Convert to ±1 labels

            self.param.update({'cate_i': cate_i, 'cate_j': 0})
            tmp_file = self.binary_model_path()

            # Model saving/loading logic
            if self.param.get('save_model_flag', 0) == 2:
                if os.path.exists(tmp_file):
                    with open(tmp_file, 'rb') as f:
                        tmp_model = pickle.load(f)
                    print(f"Loaded model from {tmp_file}")
                else:
                    tmp_model = LRE_SVMs_BinaryTrainer(self.param).train(train_ftr, tmp_trn_lbl, tmp_file)
                    with open(tmp_file, 'wb') as f:
                        pickle.dump(tmp_model, f)
            elif self.param.get('save_model_flag', 0) == 1:
                tmp_model = LRE_SVMs_BinaryTrainer(self.param).train(train_ftr, tmp_trn_lbl, tmp_file)
                with open(tmp_file, 'wb') as f:
                    pickle.dump(tmp_model, f)
            else:
                tmp_model = LRE_SVMs_BinaryTrainer(self.param).train(train_ftr, tmp_trn_lbl, tmp_file)

            weights[cate_idx] = tmp_model['weights']
            bias[cate_idx] = tmp_model['bias']

        self.model['esvm']['esvm_weights'] = weights
        self.model['esvm']['esvm_bias'] = bias
        self.model['esvm']['train_lbl'] = train_lbl
        self.model['cate_num'] = cate_num
        self.evaluate_training_accuracy(self.model, train_ftr, train_lbl)
        return self.model


    def binary_model_path(self):
        params = self.param
        return (
            f"{params['model_path']}/C{params['svm_C']}W{params['exemplar_weight']}"
            f"L{params['lambda1']}-{params['lambda2']}c{params['cate_i']}-{params['cate_j']}.model.pkl"
        )
class LRE_SVMs_BinaryTrainer:
    def __init__(self, param):
        self.param = param
        self.min_val = np.finfo(np.float32).eps

    def binary_model_path(self):
        params = self.param
        return (
            f"{params['model_path']}C{params['svm_C']}W{params['exemplar_weight']}"
            f"L{params['lambda1']}-{params['lambda2']}c{params['cate_i']}-{params['cate_j']}.model.pkl"
        )

    def train(self, src_ftr, src_lbl, tmp_file):
        pos_idx = (src_lbl == 1)
        pos_num = np.sum(pos_idx)
        neg_idx = (src_lbl == -1)
        neg_num = np.sum(neg_idx)

        # Augment features with bias term
        pos_aug = np.hstack((src_ftr[pos_idx], np.ones((pos_num, 1))))
        neg_aug = np.hstack((src_ftr[neg_idx], np.ones((neg_num, 1))))

        # Initialize parameters
        init_param = self.param.copy()
        init_param.update({'lambda1': 0, 'lambda2': 0})

        # Initialization logic
        # tmp_file = self.binary_model_path(init_param)
        if os.path.exists(tmp_file):
            with open(tmp_file, 'rb') as f:
                tmp_model = pickle.load(f)
            aug_weight = np.hstack((tmp_model['weights'], tmp_model['bias'][:, None]))
            print(f"Initialized from {tmp_file}")
        else:
            # Normalize positive samples
            norm_factor = np.maximum(np.sum(pos_aug**2, axis=1, keepdims=True), self.min_val)
            aug_weight = pos_aug / norm_factor

            # Initial optimization
            start = time()
            aug_weight, _ = SVTOptimizer(init_param).optimize(
                self.logistic_loss, aug_weight, pos_aug, neg_aug, pos_aug, src_ftr, src_lbl
            )
            # print(aug_weight.shape)
            print(f"Initial optimization took {time()-start:.2f}s")

        # Main optimization
        start = time()
        aug_weight, _ = SVTOptimizer(self.param).optimize(
            self.logistic_loss, aug_weight, pos_aug, neg_aug, pos_aug, src_ftr, src_lbl
        )
        print(f"Main optimization took {time()-start:.2f}s")

        return {
            'weights': aug_weight[:, :-1],
            'bias': aug_weight[:, -1]
        }

    def logistic_loss(self, W, pos, neg):

        max_val = np.finfo(np.float64).max / 2  # 防止数值溢出
        # 正样本损失计算 (广播机制自动处理维度对齐)
        pos_scores = np.sum(W * pos, axis=1)  # 形状 (n_pos_samples,)
        pos_loss = np.minimum(np.exp(-pos_scores), max_val)

        neg_scores = np.dot(neg, W.T)  # 形状 (n_neg_samples,)
        neg_loss = np.minimum(np.exp(neg_scores), max_val)

        obj = {
            'pos': np.log1p(pos_loss),  # 等价于 log(1 + x)
            'neg': np.log1p(neg_loss)
        }

        grad = {
            'pos': pos_loss / (1 + pos_loss),
            'neg': neg_loss / (1 + neg_loss)
        }
        return obj, grad


class SVTOptimizer:
    def __init__(self, param):
        self.param = param
        self.lambda1 = param.get('lambda1')
        self.lambda2 = param.get('lambda2')
        self.C1 = self._calculate_C1(param)
        self.mu = param.get('mu', 1.0)
        self.max_ite = param.get('max_ite', 10)
        self.max_inner_ite = param.get('max_inner_ite', 20)
        self.min_obj_val = param.get('min_obj_val', 1e-6)
        self.min_f_thred = param.get('min_f_thred', 1e-5)
        self.verbose = param.get('verbose', True)

        # 状态变量
        self.obj_val = np.inf
        self.obj_val_prev = np.inf
        self.fmat = None
        self.tr_val = 0
        self.cnvg_flag = 0

    def _calculate_C1(self, param):
        C2 = param['svm_C']
        exemplar_weight = param['exemplar_weight']
        neg_num = ...  # 需从数据获取
        return C2 * neg_num * (-exemplar_weight) if exemplar_weight < 0 else C2 * exemplar_weight

    def optimize(self, loss_func, W_init, pos_ftr, neg_ftr, trsf_ftr, src_ftr, src_lbl):
        """主优化流程"""
        pos_num, ftr_dim = pos_ftr.shape
        # print(pos_num)
        # print(ftr_dim)
        W = W_init.copy()
        self._init_fmat(W, trsf_ftr)

        for ite in range(1, self.max_ite + 1):
            update_w = self._update_weights(loss_func, W, pos_ftr, neg_ftr, trsf_ftr)
            update_f = self._update_fmat(W, trsf_ftr) if update_w else False

            if not (update_w or update_f):
                self.cnvg_flag = 1
                break

            if self._check_convergence():
                break

        # print(W, ite)
        return W, {
            'obj_val': self.obj_val,
            'iterations': ite,
            'cnvg_flag': self.cnvg_flag
        }

    def _init_fmat(self, W, trsf_ftr):
        """初始化F矩阵"""
        if self.lambda1 == 0 and self.lambda2 == 0:
            self.fmat = np.zeros_like(W @ trsf_ftr.T)
            if self.verbose:
                print('lambda=0: skip fmat initialization')
            return

        gmat = self._logistic_prob(W @ trsf_ftr.T)
        U, S, Vh = svd(gmat, full_matrices=False)
        self.tr_val = np.sum(S)
        self.fmat = gmat.copy()

    def _update_weights(self, loss_func, W, pos_ftr, neg_ftr, trsf_ftr):
        """权重更新步骤"""

        def objective(w_vec):
            W_mat = w_vec.reshape(W.shape)
            obj_loss, grad_loss = loss_func(W_mat, pos_ftr, neg_ftr)

            # 目标函数计算
            pos_term = self.C1 * np.sum(obj_loss['pos'])
            neg_term = self.param['svm_C'] * np.sum(obj_loss['neg'])
            reg_term = self.mu * np.linalg.norm(W_mat, 'fro') ** 2
            fmat_term = self.lambda2 * np.linalg.norm(self.fmat - self._logistic_prob(W_mat @ trsf_ftr.T)) ** 2
            total_obj = pos_term + neg_term + reg_term + fmat_term + self.lambda1 * self.tr_val

            # 梯度计算
            pos_grad = -self.C1 * (grad_loss['pos'][:, None] * pos_ftr)
            # print(neg_ftr.shape)
            # print(grad_loss['neg'].shape)
            neg_grad = self.param['svm_C'] * (grad_loss['neg'].T @ neg_ftr)
            reg_grad = 2 * self.mu * W_mat
            fmat_grad = 2 * self.lambda2 * (self._logistic_prob(W_mat @ trsf_ftr.T) - self.fmat) @ trsf_ftr
            total_grad = pos_grad + neg_grad + reg_grad + fmat_grad

            return total_obj, total_grad.flatten()

        # 执行优化
        res = minimize(objective, W.flatten(), method='L-BFGS-B',
                       jac=True, options={'maxiter': self.max_inner_ite})

        if res.fun < self.obj_val - self.min_obj_val:
            W_new = res.x.reshape(W.shape)
            self.obj_val_prev = self.obj_val
            self.obj_val = res.fun
            return True, W_new
        return False, W

    def _update_fmat(self, W, trsf_ftr):
        """矩阵F更新步骤"""
        gmat = self._logistic_prob(W @ trsf_ftr.T)
        try:
            U, S, Vh = svd(gmat, full_matrices=False)
            if self.lambda2 > 0:
                s_threshold = np.maximum(S - self.lambda1 / (2 * self.lambda2), 0)
            else:
                s_threshold = 0
            self.fmat = (U * s_threshold) @ Vh
            self.tr_val = np.sum(s_threshold)

            if np.mean(np.abs(self.fmat - gmat)) < self.min_f_thred:
                return False
            return True
        except np.linalg.LinAlgError:
            if self.verbose:
                print('SVD failed, keep current fmat')
            return False

    def _check_convergence(self):
        """收敛性检查"""
        delta = abs(self.obj_val_prev - self.obj_val)
        threshold = max(self.min_obj_val, 0.01 * self.obj_val)
        return delta < threshold

    def _logistic_prob(self, scores):
        """数值稳定的logistic函数"""
        scores = np.clip(scores, -500, 500)
        return 1 / (1 + np.exp(-scores))