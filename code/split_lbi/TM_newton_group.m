function [beta,s] = TM_newton_group(y,X,d1,d2,nu)
alpha = 0.1;
BETA  = 0.5;
TOLERANCE = 1e-12;
MAX_ITER = 50;
[n,p] = size(X);
p_1_U = size(d1,2);
ind_tie = find(y==0);
ind_notie = find(y==1);
e = ones(n,1);
y1 = e(y==1);
y2 = e(y==0);

X1 = [-ones(n,1),X];
X2 = [ones(n,1),X];

X11 = X1(ind_tie,:);
X12 = X2(ind_tie,:);
X13 = X1(ind_notie,:);

d11 = d1(ind_tie,:);
d12 = d2(ind_tie,:);
d13 = d1(ind_notie,:);

beta = [1;zeros(p,1)];
s = zeros(p_1_U,1);

f = @(w,s) ( -y1' * log(normcdf(X13 * w + d13 * s)) / n - ...
    y2' * log(normcdf(X12 * w + d12 * s) - normcdf(X11 * w + d11 * s)) / n + ...
    sum(s .^ 2) / (2 * nu) );
for iter = 1:MAX_ITER
    FX = f(beta,s);
    [g,H] = Hessian_TM_tie(beta,s,y,X,d1,d2,nu);
        
    dws = -H \ g;   % Newton step
    dfws = g' * dws; % Newton decrement
    if abs(dfws) < TOLERANCE
        break;
    end
    abs(dfws)
    dw = dws(1:p+1);
    ds = dws(p+2:end);
    % backtracking
    t = 1;
    while f(beta + t * dw, s + t*ds) > FX + alpha*t*dfws || (s(1) + beta(1) + t*ds(1) + t*dw(1) <= 0)
        t = BETA*t;
    end
    beta = beta + t*dw;
    s = s + t*ds;
    
end
end