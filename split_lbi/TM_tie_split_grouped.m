function obj = TM_tie_split_grouped(data,opt,group_index,verbose)
% DESCRIPTION
% Return regularization path with group sparsity under logit model.
%% Initialization of Data %%
obj.class = 'logistic split';
y = data.y;
X = data.X;
d1 = data.d1;
d2 = data.d2;
beta = data.beta_mle;
s = data.s_mle;
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

for step_cur = 1:steps_remain
    if rec_cur > opt.t_num, break; end
    %% update \beta,z and \gamma %%
    X2beta = X2*beta;
    X1beta = X1*beta;
    d2s = d2*s;
    d1s = d1*s;
    X2beta_d2s = X2beta + d2s;
    X1beta_d1s = X1beta + d1s;
    X2beta_d2s_2 = X2beta_d2s.^2;
    X1beta_d1s_2 = X1beta_d1s.^2;
    
    f2beta_s = exp(-X2beta_d2s_2/2)/sqrt(2*pi);
    f1beta_s = exp(-X1beta_d1s_2/2)/sqrt(2*pi);
    F2beta_s = normcdf(X2beta_d2s);
    F1beta_s = normcdf(X1beta_d1s);
    
    ee3 = f1beta_s ./ F1beta_s;
    ee2 = f2beta_s ./ (F2beta_s - F1beta_s);
    ee1 = f1beta_s ./ (F2beta_s - F1beta_s);
    
    ee13 = ee3(ind_notie);
    ee12 = ee2(ind_tie);
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
end