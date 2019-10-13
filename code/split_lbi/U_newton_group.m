function [beta,s] = U_newton_group(y,X,d1,d2,nu)
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

beta = [0.5;zeros(p,1)];
s = zeros(p_1_U,1);

[A,AtopA] = proj_A(X1,X2,d1,d2);

f = @(w,s) ( -y1' * log( (X13 * w + d13 * s + 1) / 2 ) / n - ...
    y2' * log( (X12 * w + d12 * s + 1) / 2 - (X11 * w + d11 * s + 1) / 2 ) / n + ...
    sum(s .^ 2) / (2 * nu) );
for iter = 1:MAX_ITER
    FX = f(beta,s);
    [g,H] = Hessian_U_tie(beta,s,y,X,d1,d2,nu);
        
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
    
    while true
        beta_tmp = beta + t * dw;
        s_tmp = s + t * ds;
        est_tmp = [beta_tmp;s_tmp];
        
        est_proj = proj_U(est_tmp,A,AtopA);
        beta_proj = est_proj(1:length(beta_tmp));
        s_proj = est_proj(length(beta_tmp)+1:end);
        if f(beta_proj, s_proj) <= FX + alpha*t*dfws
            break;
        else
            t = BETA*t;
        end
        [f(beta_proj, s_proj),FX + alpha*t*dfws]
    end
    beta = beta + t*dw;
    s = s + t*ds;
    
end
end