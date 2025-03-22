function out_param = LRE_SVMs_predict(test_ftr, test_lbl, model)
predict_val = get_predict_val(test_ftr, model);

if isfield(model,'dam_flag') && model.dam_flag
    predict_val = predict_val./repmat(max(sum(predict_val, 2),eps), 1, size(predict_val, 2));
    predict_val = predict_val /max(double(eps('single')), median(predict_val(:)));
    tid = tic;
    K_tgt = rbf_kernel(test_ftr, test_ftr, 1);
    for i=1:model.cate_num
        [~, tmp_val] = binary_dam(K_tgt, test_ftr, predict_val(:,i), model);
        predict_val(:,i) = tmp_val;
    end
    time = toc(tid);
    disp(['predict dam time: ' num2str(time)]);
else
    disp('dg: no dam in predict');
end
out_param = xmy_accuracy(predict_val, test_lbl);
out_param.model = model;
end

function predict_val = get_predict_val(test_ftr, model)
min_val = double(eps('single'));

[smpl_num, ftr_dim] = size(test_ftr);
cate_num = model.cate_num;
%trn_smpl_num = model.trn_smpl_num;

tid = tic;
esvm_predict_val = model.esvm.esvm_weights * test_ftr' + repmat(model.esvm.esvm_bias, 1, smpl_num);
time = toc(tid);
disp(['exemplar predict time : ' num2str(time) ' s ']);

esvm_predict_prob = exemplar_logistic_prob(esvm_predict_val);
clear esvm_predict_val;
esvm_predict_prob = esvm_predict_prob.*repmat(model.exemplar_prior, 1, smpl_num);
esvm_predict_p = esvm_predict_prob;

predict_val = zeros(smpl_num, model.cate_num);
for cate_i = 1:cate_num
    disp(['predict cate: ' num2str(cate_i)]);
    cate_idx = (model.esvm.train_lbl == cate_i);
    if sum(cate_idx) > model.prdct_top_num
        top_flag = true;
    else
        top_flag = false;
    end
    %tmp_prior = model.exemplar_prior(cate_idx);
    for smpl_i = 1:smpl_num
        tmp_p = esvm_predict_p(cate_idx, smpl_i);
        %tmp_prob = esvm_predict_prob(cate_idx, smpl_i);
        if top_flag
            [top_p, top_loc] = maxk(tmp_p, model.prdct_top_num);
        else
            top_p = tmp_p;
        end
        %predict_val(smpl_i, cate_i) = mean(top_p);
        predict_val(smpl_i, cate_i) = sum(top_p);
    end
end
end

function [model, pred_val] = binary_dam(K_tgt, tgt_ftr, rg_val, model)
%trn_smpl_num = size(trn_ftr, 1);
tgt_smpl_num = size(tgt_ftr, 1);
%cate_smpl_num = sum(cate_idx);
min_val = double(eps('single'));
%beta = model.exemplar_prior(cate_idx);
%beta = beta/max(sum(beta),min_val);

K = K_tgt + eye(tgt_smpl_num)/model.dam_lambda;

model = svmtrain(rg_val, [(1:tgt_smpl_num)', K], sprintf('-s 3 -t 4 -c %g -p %g -q', model.dam_C, model.dam_eps));
pred_val =  K_tgt(:,model.SVs)*model.sv_coef - model.rho;
%pred_val = y;
end

function knl = rbf_kernel(ftr1, ftr2, sigma)
%input: ftr1: n1*d, ftr2: n2*d
%output: knl n1*n2
knl = L2_distance_2(ftr1', ftr2');
%div = 2*sigma*sigma;
div = sigma*median(knl(:));
knl = exp(-knl/div);
end

function n2 = L2_distance_2(x,c,df)
if nargin < 3
    df = 0;
end
[dimx, ndata] = size(x);
[dimc, ncentres] = size(c);
if dimx ~= dimc
    error('Data dimension does not match dimension of centres')
end
n2 = (ones(ncentres, 1) * sum((x.^2), 1))' + ...
    ones(ndata, 1) * sum((c.^2),1) - ...
    2.*(x'*(c));
n2 = real(full(n2));
n2(n2<0) = 0;
if (df==1)
    n2 = n2.*(1-eye(size(n2)));
end
end

function out_param = xmy_accuracy(predict_val, tst_lbl)
[smpl_num, cate_num] = size(predict_val);
assert(length(tst_lbl) == smpl_num);
%assert(length(unique(tst_lbl)) == cate_num);
[predict_max, predict_lbl] = max(predict_val, [], 2);
%out_param.cate_prec = zeros(cate_num, 1);
out_param.cate_conf_mat = zeros(cate_num);
for cate_i = 1:cate_num
    cate_idx = (tst_lbl == cate_i);
    %  out_param.cate_prec(cate_i) = sum(predict_lbl(cate_idx) == cate_i)/sum(cate_idx);
    for cate_j = 1:cate_num
        out_param.cate_conf_mat(cate_i,cate_j) = sum(predict_lbl(cate_idx) == cate_j)/sum(cate_idx);
    end
end
%precision = mean(out_param.cate_prec);
%precision = mean(diag(out_param.cate_conf_mat));
%out_param.accuracy = precision;
out_param.accuracy =  mean(diag(out_param.cate_conf_mat));
out_param.predict_lbl = predict_lbl;
end

function prob = exemplar_logistic_prob(esvm_predict_val)
prob = 1./(1 + exp(-esvm_predict_val));
end
