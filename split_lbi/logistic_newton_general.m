function theta = logistic_newton_general(y,X,lambda,A,b)

warning off;
alpha = 0.1;
BETA  = 0.5;
TOLERANCE = 1e-12;
MAX_ITER = 50;
[n,p] = size(X);
ind_tie = find(y==0);
ind_notie = find(y==1);
e = ones(n,1);
y1 = e(y==1);
y2 = e(y==0);

A = [zeros(size(A,1),1),A];

X1 = [-ones(n,1),X];
X2 = [ones(n,1),X];

X11 = X1(ind_tie,:);
X12 = X2(ind_tie,:);
X13 = X1(ind_notie,:);
s = [1;zeros(p,1)];
f = @(w) ( y1' * log(1 + exp(-X13 * w)) / n - y2' * log(1 ./(1 + exp(-X12 * w)) - 1 ./ (1 + exp(-X11 * w))) / n + lambda * sum((A * w - b) .^ 2));
for iter = 1:MAX_ITER
    FX = f(s);
    AtopA = A' * A;
    Atopb = A' * b;
    X2s = X2*s;
    X1s = X1*s;
    exp_cs2 = exp(-X2s);
    exp_cs1 = exp(-X1s);
        
%     X11s = X11 * s;
%     X12s = X12 * s;
%     X13s = X13 * s;
    
%     exp_cs11 = exp(-X11s);
%     exp_cs12 = exp(-X12s);
%     exp_cs13 = exp(-X13s);
    
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
        
        
    g =  - X13' * ee13 / n - X12' * ee12 / n + X11' * ee11 / n + ...
        2 * lambda * AtopA * s - 2 * lambda * Atopb;
        
    ee32 = ee3./(1+exp_cs1);
    ee222 = -ee2.*((exp_cs2 - 1)./(exp_cs2 + 1)) + ee2.*ee2;
    ee221 = -ee2.*ee1;
    ee121 = ee1.*((exp_cs1 - 1)./(exp_cs1 + 1)) + ee1.*ee1;
    ee122 = -ee1.*ee2;

     
    ee32 = ee32(ind_notie);
    ee222 = ee222(ind_tie);
    ee221 = ee221(ind_tie);
    ee122 = ee122(ind_tie);
    ee121 = ee121(ind_tie);
    
    D3 = X13 .* repmat(ee32,1,p+1);
    D22 = X12 .* repmat(ee222,1,p+1);
    D21 = X12 .* repmat(ee221,1,p+1);
    D12 = X11 .* repmat(ee122,1,p+1);
    D11 = X11 .* repmat(ee121,1,p+1);
        
    H = D3' * X13 / n + D22' * X12 / n + D21' * X11 / n + D12' * X12 / n + D11' * X11 / n + AtopA * 2 * lambda;
    H = (H + H') / 2;
    ds = -pinv(full(H)) * g;   % Newton step
    dfs = g' * ds; % Newton decrement
    %abs(dfs)
    if abs(dfs) < TOLERANCE
        break;
    end
    % backtracking
    t = 1;
    while f(s + t*ds) > FX + alpha*t*dfs || (s(1) + t*ds(1) <= 0)
        t = BETA*t;
    end
    s = s + t*ds;
end
theta = s;
end
