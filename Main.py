from config import *
from data_loader import *
from LRE_SVMs_train import LRE_SVM_Trainer

if __name__ == '__main__':
    params = config
    data_root = './Office_Caltech_10/'
    datadict = preprocessing()
    src_data = select_domain_data(datadict, [0,1])
    tgt_dara = select_domain_data(datadict, [2,3])
    model = LRE_SVM_Trainer(params)
    model_dict = model.train(src_data)
    print(model_dict)
    # print(data)


