import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVR

class LRE_SVMs_Predictor:
    def __init__(self, model):
        self.model = model
        self.min_val = np.finfo(np.float32).eps

    def predict(self, test_data):
        test_ftr = test_data['features']
        test_lbl = test_data['category_labels'].flatten()
        # 获取预测值
        predict_val = self.get_predict_val(test_ftr)

        # 如果模型启用了 DAM（Domain Adaptation Model），则进行后处理
        if self.model.get('dam_flag', False):
            predict_val = self.dam_post_processing(test_ftr, predict_val)
        else:
            print("dg: no dam in predict")

        # 计算准确率等指标
        out_param = self.compute_accuracy(predict_val, test_lbl)
        out_param['model'] = self.model
        return out_param

    def get_predict_val(self, test_ftr):
        smpl_num, ftr_dim = test_ftr.shape
        cate_num = self.model['cate_num']

        # 计算 ESVM 的预测值
        esvm_weights = self.model['esvm']['esvm_weights']
        esvm_bias = self.model['esvm']['esvm_bias'].reshape(-1,1)
        esvm_predict_val = esvm_weights @ test_ftr.T + esvm_bias
        esvm_predict_prob = self.exemplar_logistic_prob(esvm_predict_val)

        # 加权预测概率
        # esvm_predict_prob *= self.model['exemplar_prior'][:, None]

        # 初始化预测值矩阵
        predict_val = np.zeros((smpl_num, cate_num))

        # 为每个类别计算预测值
        for cate_i in range(1, cate_num + 1):
            print(f"Predicting category: {cate_i}")
            cate_idx = (self.model['esvm']['train_lbl'] == cate_i)
            if np.sum(cate_idx) > self.model['prdct_top_num']:
                top_flag = True
            else:
                top_flag = False

            for smpl_i in range(smpl_num):
                tmp_p = esvm_predict_prob[cate_idx, smpl_i]
                if top_flag:
                    top_p = np.sort(tmp_p)[-self.model['prdct_top_num']:]
                else:
                    top_p = tmp_p
                predict_val[smpl_i, cate_i - 1] = np.sum(top_p)

        return predict_val

    def dam_post_processing(self, test_ftr, predict_val):
        smpl_num, _ = test_ftr.shape
        K_tgt = self.rbf_kernel(test_ftr, test_ftr, sigma=1.0)

        for i in range(self.model['cate_num']):
            _, tmp_val = self.binary_dam(K_tgt, test_ftr, predict_val[:, i])
            predict_val[:, i] = tmp_val
        return predict_val

    def binary_dam(self, K_tgt, tgt_ftr, rg_val):
        tgt_smpl_num = tgt_ftr.shape[0]
        K = K_tgt + np.eye(tgt_smpl_num) / self.model['dam_lambda']

        # 使用 SVR 模拟 MATLAB 中的 SVM 回归
        svr = SVR(C=self.model['dam_C'], epsilon=self.model['dam_eps'], kernel='precomputed')
        svr.fit(K, rg_val)
        pred_val = K_tgt @ svr.dual_coef_.T - svr.intercept_
        return svr, pred_val

    def rbf_kernel(self, ftr1, ftr2, sigma):
        knl = cdist(ftr1, ftr2, metric='sqeuclidean')
        div = sigma * np.median(knl)
        knl = np.exp(-knl / div)
        return knl

    def compute_accuracy(self, predict_val, tst_lbl):
        smpl_num, cate_num = predict_val.shape
        assert len(tst_lbl) == smpl_num

        # 获取预测标签
        predict_lbl = np.argmax(predict_val, axis=1) + 1

        # 计算混淆矩阵
        conf_mat = confusion_matrix(tst_lbl, predict_lbl, labels=np.arange(1, cate_num + 1))
        accuracy = np.mean(np.diag(conf_mat) / np.sum(conf_mat, axis=1))

        print("测试准确率: {:.2f}%".format(accuracy * 100))
        print("训练混淆矩阵:")
        print(conf_mat)

        return {
            "accuracy": accuracy,
            "cate_conf_mat": conf_mat,
            "predict_lbl": predict_lbl
        }

    @staticmethod
    def exemplar_logistic_prob(esvm_predict_val):
        return 1.0 / (1.0 + np.exp(-esvm_predict_val))