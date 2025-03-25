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
    # 假设 preprocessing() 函数已经提取了特征并保存为字典格式
    # 数据结构包含 'features', 'domain_labels', 'category_labels'
    logging.info('---加载特征---')
    data = preprocessing()

    # 配置领域索引
    train_domains = [0, 1]  # 用于训练的领域
    test_domains = [2, 3]   # 用于测试的领域

    # 根据领域索引分割数据
    train_idx = np.isin(data['domain_labels'], train_domains)
    test_idx = np.isin(data['domain_labels'], test_domains)

    X_train = data['features'][train_idx]
    y_train = data['category_labels'][train_idx]

    X_test = data['features'][test_idx]
    y_test = data['category_labels'][test_idx]

    # 数据标准化（对特征进行归一化）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 使用 SVM 训练模型
    logging.info("训练 SVM 模型...")
    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)

    # 在测试集上评估
    print("评估模型...")
    y_pred = svm.predict(X_test)

    # 输出分类报告和准确率
    logging.info("分类报告:")
    logging.info(classification_report(y_test, y_pred, target_names=Config.CATEGORIES))

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"测试集准确率: {accuracy:.2f}")

    # 输出混淆矩阵
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("混淆矩阵:")
    logging.info(conf_matrix)
