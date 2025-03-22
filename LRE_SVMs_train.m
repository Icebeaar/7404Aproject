function model = LRE_SVMs_train(data, param)

train_ftr = data.ftr;
train_lbl = data.lbl;
[trn_smpl_num, ftr_dim] = size(train_ftr);
cate_num = max(train_lbl);

weights = zeros(size(train_ftr));
bias = zeros(trn_smpl_num, 1);
for cate_i = 1:cate_num
    disp(['LRE_SVMs training... : category : ' num2str(cate_i)]);
    cate_idx = (train_lbl == cate_i);
    tmp_trn_lbl = 2 * cate_idx - 1;
    
    param.cate_i = cate_i;
    param.cate_j = 0;
    tmp_file = binary_model_path(param);
    switch param.save_model_flag
        case 2
            if exist(tmp_file, 'file')
                load(tmp_file, 'tmp_model');
                disp(['load model from: ' tmp_file]);
            else
                disp('no model: creating...');
                tmp_model = LRE_SVMs_binary_train(train_ftr, tmp_trn_lbl, param);
                save(tmp_file, 'tmp_model');
                disp(['save model to: ' tmp_file]);
            end
        case 1
            tmp_model = LRE_SVMs_binary_train(train_ftr, tmp_trn_lbl, param);
            save(tmp_file, 'tmp_model');
            disp(['save model to: ' tmp_file]);
        case 0
            tmp_model = LRE_SVMs_binary_train(train_ftr, tmp_trn_lbl, param);
        otherwise
            disp(['unknown save_model_flag: ' num2str(param.save_model_flag)]);
    end
    
    weights(cate_idx, :) = tmp_model.weights;
    bias(cate_idx) = tmp_model.bias;
end
model.esvm.esvm_weights = weights;
model.esvm.esvm_bias = bias;
model.esvm.train_lbl = train_lbl;
model.cate_num = cate_num;
model.trn_smpl_num = trn_smpl_num;
model.dam_flag = param.dam_flag;
model.prdct_top_num = param.prdct_top_num;

model.dam_C = param.dam_g1;
model.dam_lambda = param.dam_g2;
model.dam_eps = param.dam_eps;
end



function  file = binary_model_path(param)
file = [
    param.model_path...   %path
    'C' num2str(param.svm_C) ... %svm C
    'W' num2str(param.exemplar_weight) ... %exemplar weight
    'L' num2str(param.lambda1) '-' num2str(param.lambda2) ...%lambda
    'c' num2str(param.cate_i) '-' num2str(param.cate_j) ...%category
    '.model.mat'...
    ];
end


function model = LRE_SVMs_binary_train(src_ftr, src_lbl, param)

min_val = double(eps('single'));

[trn_smpl_num, ftr_dim] = size(src_ftr);
pos_idx = (src_lbl == 1);
pos_num = sum(pos_idx);
neg_idx = (src_lbl == -1);
neg_num = sum(neg_idx);

pos_aug_src_ftr = [src_ftr(pos_idx, :) ones(pos_num, 1)];
neg_aug_src_ftr = [src_ftr(neg_idx, :) ones(neg_num, 1)];
clear src_ftr;

init_param = param;
init_param.lambda1 = 0;
init_param.lambda2 = 0;

tmp_file = binary_model_path(init_param);
if exist(tmp_file, 'file')
    load(tmp_file, 'tmp_model');
    disp(['init from: ' tmp_file]);
    aug_weight = [tmp_model.weights tmp_model.bias];
else
    aug_weight = pos_aug_src_ftr./repmat(max(sum(pos_aug_src_ftr.*pos_aug_src_ftr, 2), min_val), 1, ftr_dim+1);
    tid = tic;
    [aug_weight, out_init_param] = find_min_svt(@logistic_loss, aug_weight, [], pos_aug_src_ftr, neg_aug_src_ftr, pos_aug_src_ftr, init_param);
    time = toc(tid);
    disp(['init: lambda = 0, positive samples: ' num2str(pos_num) ', W/F ite : ' num2str(out_init_param.w_f_ite), ' , objective value: ' num2str(out_init_param.obj_val) ' , time(s): ' num2str(time)]);
end

tid = tic;
[aug_weight, out_param] = find_min_svt(@logistic_loss, aug_weight, [], pos_aug_src_ftr, neg_aug_src_ftr, pos_aug_src_ftr, param);
time = toc(tid);
disp(['positive samples: ' num2str(pos_num) ', W/F ite : ' num2str(out_param.w_f_ite), ' , objective value: ' num2str(out_param.obj_val) ' , time(s): ' num2str(time)]);

model.weights = aug_weight(:, 1:ftr_dim);
model.bias = aug_weight(:, ftr_dim+1);
end

function [obj, det_vec] = weight_fun(loss_fun, weight_vec, fmat, tgt_cor, pos_src_ftr, neg_src_ftr, trsf_ftr, pos_num, ftr_dim, C1, C2, lambda2, mu)
weight = reshape(weight_vec, pos_num, ftr_dim);
[obj_loss, det_loss] = loss_fun(weight, pos_src_ftr, neg_src_ftr);
%[emat, gmat] = logistic_fmat(weight, trsf_ftr);
gmat = exemplar_logistic_prob(weight*trsf_ftr.T);
obj = weight_obj(weight, C1, C2, tgt_cor, obj_loss.pos_loss, obj_loss.neg_loss, mu) + weight_fmat_fun(gmat, fmat)*lambda2;
det = weight_det(weight, ftr_dim, C1, C2, pos_src_ftr, neg_src_ftr, tgt_cor, det_loss.pos_loss, det_loss.neg_loss, mu) + weight_fmat_det(trsf_ftr, gmat, fmat)*2*lambda2;
det_vec = det(:);
end

function obj =  weight_fmat_fun(gmat, fmat)
diff = gmat - fmat;
obj = diff(:).T*diff(:);
end

function det =  weight_fmat_det(trsf_ftr, gmat, fmat)
%det = (gmat.^2.*emat.*(gmat - fmat))*trsf_ftr;
det = (gmat.*(1-gmat).*(gmat - fmat))*trsf_ftr;
end

function obj = weight_obj(weight, C1, C2, tgt_cor, pos_loss, neg_loss, mu)
obj = mu * sum(sum(weight.*weight)) ...
    + C1 * sum(pos_loss) ...
    + C2 * sum(sum(neg_loss));
end

function det = weight_det(weight, ftr_dim,  C1, C2, pos_src_ftr, neg_src_ftr, tgt_cor, pos_loss, neg_loss, mu)
det = 2 * mu * weight ...
    - C1 * pos_src_ftr .* repmat(pos_loss, 1, ftr_dim)  ...
    + C2 * neg_loss * neg_src_ftr;
end

function [obj_loss, det_loss] = logistic_loss(aug_weight, pos_aug_src_ftr, neg_aug_src_ftr)
max_val = realmax('double')/2;
pos_loss = min(exp(-sum(aug_weight.*pos_aug_src_ftr, 2)), max_val);
neg_loss = min(exp(aug_weight*neg_aug_src_ftr.T), max_val);

obj_loss.pos_loss = log(1 + pos_loss);
obj_loss.neg_loss = log(1 + neg_loss);

det_loss.pos_loss = pos_loss./(1+pos_loss);
det_loss.neg_loss = neg_loss ./ (1 + neg_loss);
end

function [weight, out_param] = find_min_svt(loss_fun, weight, tgt_cor, pos_src_ftr, neg_src_ftr, trsf_ftr, param)
[pos_num, ftr_dim] = size(pos_src_ftr);
[neg_num, ftr_dim] = size(neg_src_ftr);
[trsf_num, ftr_dim] = size(trsf_ftr);

lambda1 = param.lambda1;
lambda2 = param.lambda2;
if lambda1 ~= 0 || lambda2 ~= 0
    lambda_flag = 1;
else
    lambda_flag = 0;
end
C2 = param.svm_C;
if param.exemplar_weight < 0
    C1 = C2 * neg_num * (-param.exemplar_weight);
else
    C1 = C2 * param.exemplar_weight;
end
mu = 1;

if lambda_flag
    fmat = exemplar_logistic_prob(weight*trsf_ftr.T);
    [tmp_u, tmp_S, tmp_v] = svd(fmat, 'econ');
    sngl_val = diag(tmp_S);
    tr_val = sum(sngl_val);
else
    fmat = zeros(pos_num, trsf_num);
    tr_val = 0;
    disp('lambda =0 : do not need fmat');
end

opt_w.Method = 'lbfgs';
opt_w.Display = 'off';
opt_w.MaxIter = param.max_inner_ite;
obj_val = inf;
obj_val_new = inf;
cnvg_flag = 0;
%ite=0;
w_min_flag = 0;
update_f_flag = 1;

weight_vec = weight(:);
if lambda2 >0
    svt_thred = lambda1/lambda2/2;
    disp(['svt thred:' num2str(svt_thred)]);
else
    svt_thred = 0;
end
for ite_i = 1:param.max_ite
    
    %% given F, update W
    update_w_flag = 0;
    if update_f_flag || w_min_flag <= 0
        tid = tic;
        [weight_vec_new, obj_val_new, w_min_flag, out_para] ...
            = minFunc(@(weight_vec)weight_fun(loss_fun, weight_vec, fmat, tgt_cor, pos_src_ftr, neg_src_ftr, trsf_ftr, pos_num, ftr_dim, C1, C2, lambda2, mu),...
            weight_vec, opt_w);
        obj_val_new = obj_val_new + lambda1 * tr_val;
        if obj_val_new < obj_val - param.min_obj_val
            weight_vec = weight_vec_new;
            obj_val = obj_val_new;
            weight = reshape(weight_vec, pos_num, ftr_dim);
            gmat = exemplar_logistic_prob(weight*trsf_ftr.T);
            update_w_flag = 1;
            time = toc(tid);
            %disp(['update W, ites: ' num2str(out_para.iterations) ' , objective value: ' num2str(obj_val) ',time(s):' num2str(time) ]);
            %ite = ite + out_para.iterations;
        end
    end %update W
    
    if ~lambda_flag
        cnvg_flag = w_min_flag > 0;
        if update_w_flag && ~cnvg_flag
            continue;
        else
            break;
        end
    end
    
    %% given W, udate F
    update_f_flag = 0;
    if update_w_flag
        tid = tic;
        try
            [tmp_u, tmp_S, tmp_v] = svd(gmat, 'econ');
            tmp_S = tmp_S - diag(svt_thred*ones(size(tmp_S, 1), 1));
            tmp_S(tmp_S < 0) = 0;
            fmat = tmp_u*tmp_S*tmp_v.T;
            tr_val = sum(diag(tmp_S));
            if mean(abs(fmat(:)-gmat(:))) < param.min_f_thred
                break;
            else
                update_f_flag = 1;
            end
        catch
            disp('svd error');
            update_f_flag = 0;
        end
        time = toc(tid);
        if update_f_flag
            %disp(['ite : ' num2str(ite_i) ', update F: svd: trace norm: ' num2str(tr_val) ', time: ' num2str(time)]);
        end
    end
    
    if ~update_w_flag && ~update_f_flag
        cnvg_flag = double(w_min_flag > 0);
        break;
    end
end

weight = reshape(weight_vec, pos_num, ftr_dim);
out_param.obj_val = obj_val;
%out_param.ttl_ite = ite;
out_param.w_f_ite = ite_i;
out_param.cnvg_flag = cnvg_flag;
end

function prob = exemplar_logistic_prob(esvm_predict_val)
prob = 1./(1 + exp(-esvm_predict_val));
end