from config import *
from data_loader import *
from LRE_SVMs_train import LRE_SVM_Trainer
from LRE_SVMs_predict import LRE_SVMs_Predictor
from sklearn.metrics import accuracy_score, confusion_matrix
import time

if __name__ == '__main__':
    params = config
    data_root = './Office_Caltech_10/'
    datadict = preprocessing()
    src_data = select_domain_data(datadict, [0,1])
    tgt_data = select_domain_data(datadict, [2,3])
    model = LRE_SVM_Trainer(params)
    start = time.time()
    model_dict = model.train(src_data)
    end = time.time()
    print(f'Training process took {end-start:.2f}s ----------')
    predictor = LRE_SVMs_Predictor(model_dict)
    predictor.predict(tgt_data)

    print(model_dict)
    # print(data)


