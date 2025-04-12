import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_loader import *
import logging
import os

if __name__ == '__main__':

    log_dir = 'log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'baseline.log')

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"  # 覆盖写入日志
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    # 加载提取的特征
    logging.info('---加载特征---')
    data = preprocessing()

    train_domains = [0, 1]  
    test_domains = [2, 3]  

    train_idx = np.isin(data['domain_labels'], train_domains)
    test_idx = np.isin(data['domain_labels'], test_domains)

    X_train = data['features'][train_idx]
    y_train = data['category_labels'][train_idx]

    X_test = data['features'][test_idx]
    y_test = data['category_labels'][test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logging.info("训练 SVM 模型...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)

    print("评估模型...")
    y_pred = svm.predict(X_test)

    logging.info("分类报告:")
    logging.info(classification_report(y_test, y_pred, target_names=Config.CATEGORIES))

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"测试集准确率: {accuracy:.2f}")

    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("混淆矩阵:")
    logging.info(conf_matrix)
