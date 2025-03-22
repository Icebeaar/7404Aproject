function kmm_weight = LRE_SVMs_mmd(model, src_ftr, tgt_ftr, param)
tmp_file = [param.model_path '\' mmd_name(param) '.mat'];
%if false
if exist(tmp_file, 'file')
    load(tmp_file, 'kmm_weight');
    disp(['load mmd prior from: ' tmp_file]);
else
    cate_num = model.cate_num;
    kmm_weight = zeros(length(model.esvm.train_lbl), 1);
    
    for cate_i = 1:cate_num
        cate_idx = (model.esvm.train_lbl == cate_i);
        cate_smpl_num = sum(cate_idx);
        if 0 == cate_smpl_num
            disp(['train prior: no exemplar for cate: ' num2str(cate_i)]);
            continue;
        end
        %cate_weights = model.esvm.esvm_weights(cate_idx, :);
        %cate_bias = model.esvm.esvm_bias(cate_idx);
        cate_prior = mmd_prior(src_ftr, tgt_ftr, [], [], cate_idx, param, model);
        kmm_weight(cate_idx) = cate_prior;
        disp(['mmd cate: ' num2str(cate_i)]);
    end
    save(tmp_file, 'kmm_weight');
    disp(['save mmd prior to: ' tmp_file]);
end
%kmm_weight = get_exemplar_prior(model);
end

function prior = get_exemplar_prior(model)
cate_num = model.cate_num;
prior = zeros(model.trn_smpl_num, 1);
for cate_i = 1:cate_num
    cate_idx = (model.esvm.train_lbl == cate_i);
    cate_smpl_num = sum(cate_idx);
    if 0 == cate_smpl_num
        disp(['train prior: no exemplar for cate: ' num2str(cate_i)]);
        continue;
    end
    prior(cate_idx) = 1.0/cate_smpl_num;
end
end

function prob = mmd_prior(trn_ftr, tgt_ftr, cate_weights, cate_bias, cate_idx, param, model)
trn_smpl_num = size(trn_ftr, 1);
tgt_smpl_num = size(tgt_ftr, 1);
cate_smpl_num = sum(cate_idx);
min_val = double(eps('single'));

src_prd_prob = trn_ftr;
tgt_prd_prob = tgt_ftr;

%%
knl1 = rbf_kernel(src_prd_prob, src_prd_prob, param.mmd_sig);
knl2 = rbf_kernel(tgt_prd_prob, tgt_prd_prob, param.mmd_sig);
knl12 = rbf_kernel(src_prd_prob, tgt_prd_prob, param.mmd_sig);

tgt_mean = mean(knl2(:));
neg_knl1 = knl1(~cate_idx,:);
neg_knl1 = neg_knl1(:,~cate_idx);
neg_mean1 = mean(neg_knl1(:));  %n-*n-
neg_knl12 = knl12(~cate_idx,:);
neg_mean12 = mean(neg_knl12(:)); %n-*tgt
pos_knl1 = knl1(cate_idx,:);
pos_neg_knl1 = pos_knl1(:,~cate_idx); %n+*n-
pos_knl1 = pos_knl1(:, cate_idx); %n+*n+
pos_knl12 = knl12(cate_idx,:); %n+*tgt
pos_num = cate_smpl_num;
neg_num = trn_smpl_num - cate_smpl_num;

dist = zeros(cate_smpl_num, 1);
%cate_idx_ids = find(cate_idx);
for i = 1:cate_smpl_num
    %pos_id = cate_idx_ids(i);
    tmp_mean1 = pos_num * pos_num *pos_knl1(i,i) + 2*pos_num*sum(pos_neg_knl1(i,:)) + neg_num * neg_num * neg_mean1;
    tmp_mean1 = tmp_mean1/trn_smpl_num/trn_smpl_num;
    tmp_mean12 = sum(neg_mean12(:)) + pos_num * sum(pos_knl12(i,:));
    tmp_mean12 = tmp_mean12/trn_smpl_num/tgt_smpl_num;
    dist(i) = tmp_mean1 + tgt_mean - 2*tmp_mean12;
end
sig = median(dist);
prob = exp(-dist/sig);
prob = prob/max(sum(prob), min_val);
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

function knl = linear_kernel(ftr1, ftr2)
%input: ftr1: n1*d, ftr2: n2*d
%output: knl n1*n2
knl = ftr1*ftr2';
end

function  nm = mmd_name(param)
nm = [...
    'C' num2str(param.svm_C) ... %svm C
    'W' num2str(param.exemplar_weight) ... %exemplar weight
    'L' num2str(param.lambda1) '-' num2str(param.lambda2) ...%lambda
    'sg' num2str(param.mmd_sig) ...
    '.mmd' ...
    ];
end