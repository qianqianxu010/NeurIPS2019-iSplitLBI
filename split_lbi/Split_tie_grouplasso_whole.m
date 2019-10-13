function lbi = Split_tie_grouplasso_whole(data,opt)
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


%% data preprocessing

% Initialize data
comp = data.comp;
y = data.y;
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




%% Training the whole train dataset  
beta_mle = logistic_newton_general(y,X,1 / 2 / opt_lbi.nu,speye(p),sparse(p,1));
data.X = X;
data.d1 = d1;
data.d2 = d2;
data.y = y;
data.beta_mle = beta_mle;
%data.s_mle = s_mle
data.s_mle = zeros((p+1)*U,1);
lbi = logistic_tie_split_grouped(data,opt_lbi,group_index);
phi = lbi.gamma;
phi(abs(phi) < 1e-6) = 0;
lbi.phi = phi;
s = lbi.s;
s_revised = s;

for i = 1:size(phi, 2)
    % Projection for computing \tilde{\beta}
    fprintf('%d\n', i);
    s_revised(:,i) = (phi(:,i) ~= 0) .* s(:,i);
end
%% record results
lbi.s_revised = s_revised;


end