function [g,H] = Hessian_U_tie(beta,s,y,X,d1,d2,nu)
[N,p] = size(X);
ind_tie = find(y==0);
ind_notie = find(y==1);
p_1_U = size(d1,2);

X1 = [-ones(N,1),X];
X2 = [ones(N,1),X];

X11 = X1(ind_tie,:);
X12 = X2(ind_tie,:);
X13 = X1(ind_notie,:);

d11 = d1(ind_tie,:);
d12 = d2(ind_tie,:);
d13 = d1(ind_notie,:);


X2beta = X2*beta;
X1beta = X1*beta;
d2s = d2*s;
d1s = d1*s;
X2beta_d2s = X2beta + d2s;
X1beta_d1s = X1beta + d1s;


f2beta_s = 1 / 2 * ones(size(X1,1),1);
f1beta_s = 1 / 2 * ones(size(X2,1),1);
F2beta_s = (X2beta_d2s + 1) / 2;
F1beta_s = (X1beta_d1s + 1) / 2;

ee3 = f1beta_s ./ F1beta_s;
ee2 = f2beta_s ./ (F2beta_s - F1beta_s);
ee1 = f1beta_s ./ (F2beta_s - F1beta_s);

ee13 = ee3(ind_notie);
ee12 = ee2(ind_tie);
ee11 = ee1(ind_tie);

g_beta = - X13' * ee13 / N - X12' * ee12 / N + X11' * ee11 / N;
g_s =  - d13' * ee13 / N - d12' * ee12 / N + d11' * ee11 / N + s / nu;
g = [g_beta;g_s];
        

ee32 = ee3 .^ 2;
ee222 = ee2 .^ 2;
ee221 = -ee2 .* ee1;
ee121 = ee1 .^ 2;
ee122 = -ee1 .* ee2;

     
ee32 = ee32(ind_notie);
ee222 = ee222(ind_tie);
ee221 = ee221(ind_tie);
ee122 = ee122(ind_tie);
ee121 = ee121(ind_tie);
    
X3_top = X13 .* repmat(ee32,1,p+1);
X22_top = X12 .* repmat(ee222,1,p+1);
X21_top = X12 .* repmat(ee221,1,p+1);
X12_top = X11 .* repmat(ee122,1,p+1);
X11_top = X11 .* repmat(ee121,1,p+1);

H_beta = X3_top' * X13 / N + X22_top' * X12 / N + X21_top' * X11 / N + ...
    X12_top' * X12 / N + X11_top' * X11 / N; 
H_beta = (H_beta + H_beta') / 2;

D3_top = d13' * sparse(diag(ee32));
D22_top = d12' * sparse(diag(ee222));
D21_top = d12' * sparse(diag(ee221));
D12_top = d11' * sparse(diag(ee122));
D11_top = d11' * sparse(diag(ee121));

H_s = D3_top * d13 / N + D22_top * d12 / N + D21_top * d11 / N + ...
    D12_top * d12 / N + D11_top * d11 / N + speye(p_1_U) / nu;
H_s = (H_s + H_s') / 2;

H_beta_s = X3_top' * d13 / N + X22_top' * d12 / N + X21_top' * d11 / N + ...
    X12_top' * d12 / N + X11_top' * d11 / N; 

H_s_beta = H_beta_s';


H = [[H_beta,H_beta_s];[H_s_beta,H_s]];
H = (H + H')/2;
end