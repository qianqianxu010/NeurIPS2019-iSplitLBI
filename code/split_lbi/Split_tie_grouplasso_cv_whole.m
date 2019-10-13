function result = Split_tie_grouplasso_cv_whole(data,opt,cv_para,micro_ornot)
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
    cv_para.seed_cross = 1;
end
if nargin < 4 || isempty(micro_ornot), micro_ornot = true;end

if micro_ornot == true
    result{1} = 'Micro-F1';
else
    result{1} = 'Macro-F1';
end

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
seed_cross = cv_para.seed_cross;

% Initialization of X
X = sparse(N,p);
for i=1:p
    X(comp(:,1) == cand_id(i),i) = 1;
    X(comp(:,2) == cand_id(i),i) = -1;
end


%% Train-Test Split
d1 = sparse(N,(p+1)*U);
d2 = sparse(N,(p+1)*U);
count = 0;
d1_eig = 0;
for u = 1:U
    ind_u = find(user == user_id(u));
    X_tmp = X(ind_u,:);
    d1_tmp = [-ones(size(X_tmp,1),1),X_tmp];
    d1_tmptopd1_tmp = d1_tmp' * d1_tmp;
    d1_tmp_eig = norm(full(d1_tmptopd1_tmp));
    if d1_tmp_eig > d1_eig
        d1_eig = d1_tmp_eig;
    end
    d1(count+1:count+length(ind_u),(u-1)*(p+1) + 1:u*(p+1)) = d1_tmp;
    d2(count+1:count+length(ind_u),(u-1)*(p+1) + 1:u*(p+1)) = [ones(size(X_tmp,1),1),X_tmp];
   
    count = count + length(ind_u);
end



%% Setting t0 and delta for Split LBI
%dtopd = d1_train' * d1_train;
normd = d1_eig;
normX = norm(full(X'*X),2);
opt_lbi = opt;
nu = opt.nu;
delta = 1 / opt.kappa / (1 + normd / N + normX / N + 1 / opt.nu);
[~,s_mle] = logistic_newton_group(y,X,d1,d2,opt.nu);
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



%% Cross Split on whole dataset
for kkk=1:kfolds
    ind_train{kkk} = [];
end

rng(seed_cross);
for u = 1:U
    ind_u = find(user == user_id(u));
    y_u = y(ind_u);
    cp_train = cvpartition(y_u,'k',kfolds);
    for kkk = 1:kfolds
        ind_test_tmp = ind_u(test(cp_train,kkk));
        ind_train_tmp = ind_u(~ismember(ind_u,ind_test_tmp));
        ind_train{kkk} = [ind_train{kkk};ind_train_tmp];
        ind_test{kkk}{u} = ind_test_tmp;
    end
end

%% Cross validation using Split LBI
result_mxcro_dense = zeros(opt_lbi.t_num,1);
result_mxcro_sparse = zeros(opt_lbi.t_num,1);
for kkk=1:kfolds
    
    [result_mxcro_dense_tmp,result_mxcro_sparse_tmp,lbi] = ...
                        Split_tie_grouplasso_onetime(X,d1,d2,y,y_flip,...
                              opt_lbi,ind_train{kkk},ind_test{kkk},...
                              group_index,micro_ornot,opt_lbi.class);
    
    result_mxcro_dense = result_mxcro_dense + result_mxcro_dense_tmp;
    result_mxcro_sparse = result_mxcro_sparse + result_mxcro_sparse_tmp;
    result_cv{kkk} = lbi;
end
[~,ind_t_sparse] = max(result_mxcro_sparse);
[~,ind_t_dense] = max(result_mxcro_dense);

%% Training the whole train dataset  
beta_mle = logistic_newton_general(y,X,0,speye(p),sparse(p,1));
data.X = X;
data.d1 = d1;
data.d2 = d2;
data.y = y;
data.beta_mle = beta_mle;
%data.s_mle = s_mle;
data.s_mle = zeros((p+1)*U,1);
lbi = logistic_tie_split_grouped(data,opt_lbi,group_index);
phi = lbi.gamma;
phi(abs(phi) < 1e-6) = 0;
beta = lbi.beta;
s = lbi.s;
s_revised = s;
test_mxcro_dense_vec = zeros(size(phi,2),1);
test_mxcro_sparse_vec = zeros(size(phi,2),1);
for i = 1:size(phi, 2)
    % Projection for computing \tilde{\beta}
    fprintf('%d\n', i);
    s_revised(:,i) = (phi(:,i) ~= 0) .* s(:,i);
    test_mxcro_sparse_vec(i) = mxcro_compute(ind_test{kkk},y,y_flip,X,d1,d2,beta(:,i),s_revised(:,i),micro_ornot);
    test_mxcro_dense_vec(i) = mxcro_compute(ind_test{kkk},y,y_flip,X,d1,d2,beta(:,i),s(:,i),micro_ornot);
end
lbi.s_revised = s_revised;
gamma = phi(:,ind_t_sparse);
%% record results
test_mxcro_sparse = test_mxcro_sparse_vec(ind_t_sparse);
test_mxcro_dense = test_mxcro_dense_vec(ind_t_dense);
result{2} = result_cv;
result{3} = test_mxcro_sparse;
result{4} = test_mxcro_dense;
result{5} = test_mxcro_sparse_vec;
result{6} = test_mxcro_dense_vec;
result{7} = lbi;
result{8} = gamma;
result{9} = result_mxcro_sparse;
result{10} = result_mxcro_dense;
end