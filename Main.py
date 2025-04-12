from config import *
from data_loader import *
from LRE_SVMs_train import LRE_SVM_Trainer
from LRE_SVMs_predict import LRE_SVMs_Predictor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_loader import *
import logging
import os
import time

if __name__ == '__main__':
    params = config
    data_root = './Office_Caltech_10/'
    data = preprocessing()
    # src_data = select_domain_data(data, [0,1])
    # tgt_data = select_domain_data(data, [2,3])

    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'lre_svm.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"  # 覆盖写入日志
    )

    # X_train = src_data['features']
    # y_train = src_data['category_labels']
    #
    # X_test = tgt_data['features']
    # y_test = tgt_data['category_labels']
    train_domains = [0, 1] 
    test_domains = [2, 3] 

    train_idx = np.isin(data['domain_labels'], train_domains)
    test_idx = np.isin(data['domain_labels'], test_domains)

    src_data = {
        'features': data['features'][train_idx],
        'category_labels': data['category_labels'][train_idx],
        'domain_labels': data['domain_labels'][train_idx]
    }

    tgt_data = {
        'features': data['features'][test_idx],
        'category_labels': data['category_labels'][test_idx],
        'domain_labels': data['domain_labels'][test_idx]
    }

    # logging.info("对特征进行标准化...")
    # scaler = StandardScaler()
    # # src_data['features'] = scaler.fit_transform(src_data['features'])
    # # tgt_data['features'] = scaler.fit_transform(tgt_data['features'])


    model = LRE_SVM_Trainer(params)
    start = time.time()
    model_dict = model.train(src_data)
    end = time.time()

    logging.info(f'Training process took {end-start:.2f}s ----------')
    predictor = LRE_SVMs_Predictor(model_dict)
    out_param = predictor.predict(tgt_data)

    acc = out_param['accuracy']
    conf_mat = out_param['cate_conf_mat']
    logging.info(f'测试准确率：{acc:.2f}')
    logging.info(f'混淆矩阵：')
    logging.info(conf_mat)


