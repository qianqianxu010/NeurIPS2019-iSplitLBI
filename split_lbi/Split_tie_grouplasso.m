function result = Split_tie_grouplasso(data,opt,cv_para,micro_ornot)
%% data preprocessing
if ~exist('data','var'), error('No input data!'); end
if ~isfield(data,'comp')
    error('The comp is missing!');
end
if ~isfield(data,'y')
    error('The y is missing!');
end
if ~isfield(data,'user')
    error('The user is missing!');
end 

if nargin < 2 || isempty(opt), opt = [];end
if nargin < 3 || isempty(cv_para)
    cv_para.kfolds = 5;
    cv_para.test_per = 0.2;
    cv_para.seed_test = 1;
    cv_para.seed_cross = 1;
    cv_para.class = 'BT';
end
if nargin < 4 || isempty(micro_ornot), micro_ornot = true;end

% if micro_ornot == true
%     result{1} = 'Micro-F1';
% else
%     result{1} = 'Macro-F1';
% end

%% data preprocessing

% Initialize data
comp = data.comp;
y = data.y;
y_flip = data.y_flip;
user = data.user;
user_id = unique(user);
cand_id = unique([comp(:,1);comp(:,2)]);

% Initialze Number of samples, users and candidates
N = size(comp,1);
U = length(user_id);
p = length(cand_id);

% Initialize Group index
group_index = zeros(U * (p+1),1);
for u = 1:U
    group_index((u-1)*(p+1)+1:u*(p+1)) = u;
end

% Initialization of parameters for cross-validation
kfolds = cv_para.kfolds;
test_per = cv_para.test_per;
seed_test = cv_para.seed_test;
seed_cross = cv_para.seed_cross;

% Initialization of X
X = sparse(N,p);
for i=1:p
    X(comp(:,1) == cand_id(i),i) = 1;
    X(comp(:,2) == cand_id(i),i) = -1;
end


%% Train-Test Split
rng(seed_test);
d1 = sparse(N,(p+1)*U);
d2 = sparse(N,(p+1)*U);
d1_train = [];
d2_train = [];
X_train = [];
y_train = [];
y_flip_train = [];
count = 0;
d1_eig = 0;
for u = 1:U
    ind_u = find(user == user_id(u));
    y_u = y_flip(ind_u);
    y_flip_u = y_flip(ind_u);
    cp_u_train_test = cvpartition(y_u,'holdout',test_per);
    ind_train{u} = ind_u(cp_u_train_test.training);
    ind_test{u} = ind_u(cp_u_train_test.test);
    X_tmp = X(ind_u,:);
    X_train_tmp = X(ind_train{u},:);
    X_train = [X_train;X_train_tmp];
    y_train = [y_train;y(ind_train{u})];
    y_flip_train = [y_flip_train;y_flip(ind_train{u})];
    d1_tmp = [-ones(size(X_tmp,1),1),X_tmp];
    d1_tmptopd1_tmp = d1_tmp' * d1_tmp;
    d1_tmp_eig = norm(full(d1_tmptopd1_tmp));
    if d1_tmp_eig > d1_eig
        d1_eig = d1_tmp_eig;
    end
    d1(count+1:count+length(ind_u),(u-1)*(p+1) + 1:u*(p+1)) = d1_tmp;
    d1_train = [d1_train;d1(ind_train{u},:)];
    d2(count+1:count+length(ind_u),(u-1)*(p+1) + 1:u*(p+1)) = [ones(size(X_tmp,1),1),X_tmp];
    d2_train = [d2_train;d2(ind_train{u},:)];
    count = count + length(ind_u);
end

d1_train = sparse(d1_train);
d2_train = sparse(d2_train);

%% Setting t0 and delta for Split LBI
%dtopd = d1_train' * d1_train;
normd = d1_eig;
normX = norm(full(X_train'*X_train),2);
opt_lbi = opt;
nu = opt.nu;
delta = 1 / opt.kappa / (1 + normd / N + normX / N + 1 / opt.nu);
if opt_lbi.if_colledge
    delta = delta / 2;
end
if strcmp(cv_para.class,'BT')
    [beta_mle,s_mle] = logistic_newton_group(y_train,X_train,d1_train,d2_train,opt.nu);
elseif strcmp(cv_para.class,'TM')
    [beta_mle,s_mle] = TM_newton_group(y_train,X_train,d1_train,d2_train,opt.nu);
else
    [beta_mle,s_mle] = U_newton_group(y_train,X_train,d1_train,d2_train,opt.nu);
end


t0 = Inf;
for u=1:U
    t_new = nu / norm(s_mle((u-1)*(p+1)+1:u*(p+1)));
    if t_new < t0
        t0 = t_new;
    end
end
opt_lbi.t0 = t0;
opt_lbi.t_num = 500;
opt_lbi.t_seq = logspace(log10(t0), log10(t0 * opt_lbi.t_ratio), opt_lbi.t_num);
opt_lbi.delta = delta;


%% Cross Split on Train dataset
rng(seed_cross);
for kkk = 1:kfolds
    ind_train_train{kkk} = [];
end
for u = 1:U
    ind_tmp = ind_train{u};
    y_tmp = y_flip(ind_train{u});
    cp_train = cvpartition(y_tmp,'k',kfolds);
    for kkk = 1:kfolds
        ind_train_test_tmp = ind_tmp(test(cp_train,kkk));
        ind_train_train_tmp = ind_tmp(~ismember(ind_tmp,ind_train_test_tmp));
        ind_train_train{kkk} = [ind_train_train{kkk};ind_train_train_tmp];
        ind_train_test{kkk}{u} = ind_train_test_tmp;
    end
end

%% Cross validation using Split LBI
result_mxcro_dense = zeros(opt_lbi.t_num,1);
result_mxcro_sparse = zeros(opt_lbi.t_num,1);
for kkk=1:kfolds
    
    [result_mxcro_dense_tmp,result_mxcro_sparse_tmp,lbi] = ...
                        Split_tie_grouplasso_onetime(X,d1,d2,y,y_flip,...
                              opt_lbi,ind_train_train{kkk},ind_train_test{kkk},...
                              group_index,micro_ornot,cv_para.class);
    
    result_mxcro_dense = result_mxcro_dense + result_mxcro_dense_tmp;
    result_mxcro_sparse = result_mxcro_sparse + result_mxcro_sparse_tmp;
    result_cv{kkk} = lbi;
end
[~,ind_t_sparse] = max(result_mxcro_sparse);
[~,ind_t_dense] = max(result_mxcro_dense);

%% Training the whole train dataset  
data.X = X_train;
data.d1 = d1_train;
data.d2 = d2_train;
data.y = y_train;
data.beta_mle = beta_mle;
data.s_mle = s_mle;
if strcmp(cv_para.class,'BT')
    lbi = logistic_tie_split_grouped(data,opt_lbi,group_index);
elseif strcmp(cv_para.class,'TM')
    lbi = TM_tie_split_grouped(data,opt_lbi,group_index);
else
    lbi = U_tie_split_grouped(data,opt_lbi,group_index);
end
phi = lbi.gamma;
phi(abs(phi) < 1e-6) = 0;
beta = lbi.beta;
s = lbi.s;
s_revised = s;
test_mxcro_dense_vec = zeros(size(phi,2),1);
test_mxcro_sparse_vec = zeros(size(phi,2),1);
for i = 1:size(phi, 2)
    % Projection for computing \tilde{\beta}
    %fprintf('%d\n', i);
    s_revised(:,i) = (phi(:,i) ~= 0) .* s(:,i);
    test_mxcro_sparse_vec(i) = mxcro_compute(ind_test,y,y_flip,X,d1,d2,beta(:,i),s_revised(:,i),micro_ornot);
    test_mxcro_dense_vec(i) = mxcro_compute(ind_test,y,y_flip,X,d1,d2,beta(:,i),s(:,i),micro_ornot);
end
gamma = phi(:,ind_t_dense);
[pre_n1,pre_0,pre_p1,rec_n1,rec_0,rec_p1] = precision_recall(ind_test,y,y_flip,X,d1,d2,beta(:,ind_t_dense),s(:,ind_t_dense));
%% record results
result.y = y;
result.y_flip = y_flip;
result.X = X;
result.d1 = d1;
result.d2 = d2;
result.ind_test = ind_test;
test_mxcro_sparse = test_mxcro_sparse_vec(ind_t_sparse);
test_mxcro_dense = test_mxcro_dense_vec(ind_t_dense);
%result.result_cv = result_cv;
result.test_mxcro_sparse = test_mxcro_sparse;
result.test_mxcro_dense = test_mxcro_dense;
result.test_mxcro_sparse_vec = test_mxcro_sparse_vec;
result.test_mxcro_dense_vec = test_mxcro_dense_vec;
result.lbi = lbi;
result.gamma = gamma;
result.result_mxcro_sparse = result_mxcro_sparse;
result.result_mxcro_dense = result_mxcro_dense;
result.pre_rec = [pre_n1,pre_0,pre_p1,rec_n1,rec_0,rec_p1];

%% record result (c,p)
result.ps_dense_t = reshape(lbi.s(:,ind_t_dense),p+1,U);
result.ps_sparse_t = reshape(s_revised(:,ind_t_sparse),p+1,U);
result.cs_t = lbi.beta(:,ind_t_dense); % common cs
result.ps_supp = find(sum(result.cs_t~=0))'; % outlier id
end
      


    
    
    





