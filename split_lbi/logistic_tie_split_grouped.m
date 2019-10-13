function obj = logistic_tie_split_grouped(data,opt,group_index,verbose)
% DESCRIPTION
% Return regularization path with group sparsity under logit model.
%% Initialization of Data %%
obj.class = 'logistic split';
y = data.y;
X = data.X;
d1 = data.d1;
d2 = data.d2;
beta = data.beta_mle;
%beta = zeros(length(data.beta_mle),1);
%beta(1) = 1;
s = data.s_mle;
%s = zeros(length(data.s_mle),1);
[n, p] = size(X);
p_1_u = length(s);
%% Initialization of Parameter %%
opt = initial(opt);
if opt.normalize == true, X = normc(X);end
kappa = opt.kappa;
delta = opt.delta;
%% Initialization of \nu %%
if isempty(opt.nu)
    nu = n * norm(full(D), 2)^2 / norm(full(bsxfun(@minus, X, mean(X))), 2)^2; 
else
    nu = opt.nu;
end
if nargin < 4, verbose = true; end


z = zeros(p_1_u, 1);
gamma = zeros(p_1_u, 1);
obj.beta = repmat(beta, 1, opt.t_num);
obj.s = repmat(s, 1, opt.t_num);
obj.z = zeros(p_1_u, opt.t_num);
obj.gamma = zeros(p_1_u, opt.t_num);
obj.beta_tilde = repmat(beta, 1, opt.t_num);
obj.cost = zeros(1, opt.t_num);

ind_tie = find(y==0);
ind_notie = find(y==1);
X1 = [-ones(n,1),X];
X2 = [ones(n,1),X];
X11 = X1(ind_tie,:);
X12 = X2(ind_tie,:);
X13 = X1(ind_notie,:);
d11 = d1(ind_tie,:);
d12 = d2(ind_tie,:);
d13 = d1(ind_notie,:);


%% The regularization path from 0 to t0 %%

if ~opt.fast_init
    beta = zeros(p+1,1);
    s = zeros(p_1_u,1);
    t0 = 0;
else
    t0 = opt.t0;
end
rec_cur = sum(opt.t_seq <= t0) + 1;
steps_remain = ceil((opt.t_seq(end) - t0) / delta);
fprintf('The number of whole iteration %d\n',steps_remain);
%% Starting Iteration %%
G = length(unique(group_index));
if verbose, fprintf(['Linearized Bregman Iteration (', obj.class, '):\n']); end
tic
var_hist = [];
var_order = [];
for step_cur = 1:steps_remain
    if rec_cur > opt.t_num, break; end
    %% update \beta,z and \gamma %%
    X2beta = X2 * beta;
    X1beta = X1 * beta;
    d2s = d2 * s;
    d1s = d1 * s;
    exp_cs2 = exp(-X2beta - d2s);
    exp_cs1 = exp(-X1beta - d1s);
        
    F1s = 1 ./ (1 + exp_cs1);
    F2s = 1 ./ (1 + exp_cs2);
    F3s = 1 ./ (1 + exp_cs1);
        
    f1s = (F1s.^ 2) .* exp_cs1;
    f2s = (F2s.^ 2) .* exp_cs2;
    f3s = F3s .* exp_cs1;  
        
    ee3 = f3s;
    ee13 = ee3(ind_notie);
    ee2 = f2s ./ (F2s - F1s);
    ee12 = ee2(ind_tie);
    ee1 = f1s ./ (F2s - F1s);
    ee11 = ee1(ind_tie);
        
    %% update \beta
    d_beta = - X13' * ee13 / n - X12' * ee12 / n + X11' * ee11 / n;
    beta = beta - kappa * delta * d_beta;
    lambda_beta = beta(1);
    beta(1) = lambda_beta * (lambda_beta > 0);
    %% update s
    d_s = - d13' * ee13 / n - d12' * ee12 / n + d11' * ee11 / n + (s - gamma) / nu;
    s = s - kappa * delta * d_s;
    lambda_s = s([1:p+1:p_1_u]);
    s([1:p+1:p_1_u]) = lambda_s .* (lambda_s + beta(1) > 0);
    
    
    %% update z
    d_gamma = (gamma - s) / nu;
    z = z - delta * d_gamma;
%     for g = 1:G
%         g_ind = find(group_index == g);
%         gamma(g_ind) = kappa * max(0, 1 - 1 / norm(z(g_ind))) * z(g_ind);
%     end
    z_new = reshape(z,length(z)/G,G);
    gamma_mat = kappa * repmat(max(0, 1 - 1 ./ sqrt(sum(z_new.*z_new))),size(z_new,1),1) .* z_new;
    gamma_mat(abs(gamma_mat)<1e-6) = 0;
    
    gamma_nonzero = find(sum(gamma_mat~=0))';
    if sum(~ismember(gamma_nonzero,var_hist)) >= 1
        append_ind = gamma_nonzero(~ismember(gamma_nonzero,var_hist));
        var_hist = [var_hist;append_ind];
        var_order = [var_order;step_cur * ones(length(append_ind),1)];
    end
    
    
    
    gamma = reshape(gamma_mat,G * size(gamma_mat,1),1);
    
    %% Recording some of estimations in the regularization path %%
    
    while true
        dt = step_cur * delta + t0 - opt.t_seq(rec_cur);
        if dt < 0, break; end
        z_last = z + dt * d_gamma;
        obj.z(:, rec_cur) = z_last;
%         for g = 1:G
%             g_ind = find(group_index == g);
%             obj.gamma(g_ind,rec_cur) = kappa * max(0, 1 - 1 / norm(obj.z(g_ind,rec_cur))) * obj.z(g_ind,rec_cur);
%         end
        z_last_new = reshape(z_last,length(z_last)/G,G);
        gamma_last_mat = kappa * repmat(max(0, 1 - ...
            1 ./ sqrt(sum(z_last_new.*z_last_new))),size(z_new,1),1) .* z_new;
        obj.gamma(:,rec_cur) = reshape(gamma_last_mat,G * size(gamma_last_mat,1),1);
        obj.beta(:, rec_cur) = beta + kappa * dt * d_beta;
        obj.s(:, rec_cur) = s + kappa * dt * d_s;
        obj.cost(rec_cur) = toc * 1;
        rec_cur = rec_cur + 1;
        if rec_cur > opt.t_num, break; end
    end
    if verbose && ismember(step_cur, round(steps_remain ./ [100 50 20 10 5 2 4/3 20/19 1]))
        fprintf('Process: %0.2f%%. Time: %f\n', step_cur / steps_remain * 100, toc);
    end
end
fprintf('\n');
obj.nu = nu;
obj.delta = delta;
obj.t_seq = opt.t_seq;
obj.K = length(opt.t_seq);
obj.var_hist = var_hist;
obj.var_order = var_order;
end
