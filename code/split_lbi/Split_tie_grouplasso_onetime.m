function [mxcro_dense,mxcro_sparse,lbi] = Split_tie_grouplasso_onetime(X,d1,d2,y,y_flip,opt_lbi,index_train,index_test,group_index,micro_ornot,opt_class)

%% Training data
X_train = X(index_train,:);
d1_train = d1(index_train,:);
d2_train = d2(index_train,:);
y_train = y(index_train);
p = size(X_train,2);
p_1_u = size(d1_train,2);

%% Setting parameters for split lbi (with grouped)
data.X = X_train;
data.d1 = d1_train;
data.d2 = d2_train;
data.y = y_train;

if strcmp(opt_class,'BT')
    %[beta_mle,s_mle] = logistic_newton_group(y_train,X_train,d1_train,d2_train,opt_lbi.nu);
    beta_mle = logistic_newton_general(y_train,X_train,0,speye(p),sparse(p,1));
elseif strcmp(opt_class,'TM')
    [beta_mle,s_mle] = TM_newton_group(y_train,X_train,d1_train,d2_train,opt_lbi.nu);
else
    [beta_mle,s_mle] = U_newton_group(y_train,X_train,d1_train,d2_train,opt_lbi.nu);
end
data.beta_mle = beta_mle;
%data.s_mle = s_mle;
data.s_mle = zeros(p_1_u,1);
%% Implementation of Split LBI 
if strcmp(opt_class,'BT')
    lbi = logistic_tie_split_grouped(data,opt_lbi,group_index);
elseif strcmp(opt_class,'TM')
    lbi = TM_tie_split_grouped(data,opt_lbi,group_index);
else
    lbi = U_tie_split_grouped(data,opt_lbi,group_index);
end

%% compute s and s_revised
phi = lbi.gamma;
phi(abs(phi) < 1e-6) = 0;
beta = lbi.beta;
s = lbi.s;
s_revised = s;
mxcro_dense = zeros(size(phi,2),1);
mxcro_sparse = zeros(size(phi,2),1);
for i = 1:size(phi, 2)
    % Projection for computing \tilde{\beta}
    %fprintf('%d\n', i);
    s_revised(:,i) = (phi(:,i) ~= 0) .* s(:,i);
    mxcro_sparse(i) = mxcro_compute(index_test,y,y_flip,X,d1,d2,beta(:,i),s_revised(:,i),micro_ornot);
    mxcro_dense(i) = mxcro_compute(index_test,y,y_flip,X,d1,d2,beta(:,i),s(:,i),micro_ornot);
    
end

end
