addpath('.\lib\minFunc');
addpath('.\lib\minFunc\compiled');
addpath('.\lib\MinMaxSelection');
addpath('.\lib\libsvm-3.17\matlab')

%% parameters
%dam_flag:
%0: domain generalization
param.dam_flag = 0;


%save_model_flag:
%0: train model without save
%1: train & save
%2: load model from model_path
param.save_model_flag = 2;
param.model_path ='.\model\'';
if ~exist(param.model_path, 'dir')
    mkdir(param.model_path);
    disp(['folder created: ' param.model_path])
end

%parameters
param.svm_C = 0.001; 
param.lambda1 = 1; 
param.lambda2 = 4; 
param.exemplar_weight = 10; 
param.prdct_top_num = 5; 

%dam parameters
param.mmd_sig = 1;
param.dam_sig = 1;
param.dam_g1 = 1;
param.dam_g2 = 1;
param.dam_eps = 1e-3;

%optimization parameters
param.max_ite = 10;
param.max_inner_ite = 100;
param.min_f_thred = 1e-7;
param.min_obj_val = 1e-7;
param.dam_min_stop = 1e-5;

%% load data
data_file = '.\data\office_caltech_dl_ms0.mat';
load(data_file, 'ms_data');
disp(['load from: ' data_file]);
disp(['domain: ' num2str(ms_data.dm_num) ', category: ' num2str(ms_data.cate_num) ', sample: ' num2str(ms_data.smpl_num)]);

%source domain: amazon, caltech, target domain: dslr, webcam
src_dm = [1 2];
disp(' ');
disp('src dm: ');
src_data = select_dm_data(ms_data, src_dm);
if isempty(src_data)
    disp(['warning: no src data, domain lbl : ' num2str(src_dm)]);
end
tgt_dm = [3 4];
disp(' ');
disp('tgt dm: ');
tgt_data = select_dm_data(ms_data, tgt_dm);
if isempty(tgt_data)
    disp(['warning: no tgt data, domain lbl : ' num2str(tgt_dm)]);
end

disp(' ')
disp(['start: ' num2str(clock)]);
disp(' ')

%% training
disp(' ');
disp('training...')
tid =tic;
model = LRE_SVMs_train(src_data, param);
time = toc(tid);
disp(['LRE_SVMs train time: ' num2str(time) ' s']);

if model.dam_flag
    model.exemplar_prior = LRE_SVMs_mmd(model, src_data.ftr, tgt_data.ftr, param);
else
    model.exemplar_prior = ones(model.trn_smpl_num, 1);
end

%% testing
disp(' ');
disp('testing...')
tid = tic;
out_param = LRE_SVMs_predict(tgt_data.ftr, tgt_data.lbl, model);
time = toc(tid);
disp(['predict time : ' num2str(time) ' s ']);

disp(['ACC: ' num2str(out_param.accuracy)]);

disp(' ');
disp(['end: ' num2str(clock)]);

function dm_data = select_dm_data(ms_data, dm_ids)
dm_idx = false(size(ms_data.dm_lbl));
for d_i = 1:length(dm_ids)
    dm_idx(ms_data.dm_lbl == dm_ids(d_i)) = true;
end

if 0 == sum(dm_idx)
    disp(['empty domain : ' num2str(dm_ids)]);
end

ftr = ms_data.ftr(dm_idx, :);
lbl = ms_data.cate_lbl(dm_idx);

lbl_num = length(unique(lbl));
assert(max(lbl) == lbl_num);
[smpl_num, smpl_dim] = size(ftr);
assert(length(lbl) == smpl_num);

disp(['select domains: ' num2str(dm_ids) ', labels: ' num2str(lbl_num) ', samples: ' num2str(smpl_num) ', dim: ' num2str(smpl_dim)]);

dm_data.ftr = ftr;
dm_data.lbl = lbl;
dm_data.idx = dm_idx;
dm_data.dm_lbl = ms_data.dm_lbl(dm_idx);

dm_data.lbl_num = lbl_num;
dm_data.ftr_dim = ms_data.ftr_dim;
dm_data.smpl_num = smpl_num;
end
