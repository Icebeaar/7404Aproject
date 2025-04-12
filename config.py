import os
from dataclasses import dataclass

config = {
    'dam_flag': 0,  
    'save_model_flag': 2,  
    'model_path': r'./model',  

    'svm_C': 0.001, 
    'lambda1': 1.0, 
    'lambda2': 4.0, 
    'exemplar_weight': 10.0,
    'prdct_top_num': 5, 

    'mmd_sig': 1.0, 
    'dam_sig': 1.0, 
    'dam_g1': 1.0,  
    'dam_g2': 1.0,
    'dam_eps': 1e-3, 
    'dam_min_stop': 1e-5, 

    'max_ite': 10, 
    'max_inner_ite': 100, 
    'min_f_thred': 1e-7,
    'min_obj_val': 1e-7, 
    # 'dam_min_stop': 1e-5 
}
