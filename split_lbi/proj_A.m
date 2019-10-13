function [A,AtopA] = proj_A(X1,X2,d1,d2)
p_1 = size(X1,2);
p_1_u = size(d1,2);
U = round(p_1_u / p_1);
Xd_non = sparse(U,p_1 + p_1_u);
Xd_non(:,1) = 1;
ind = p_1 * [1:U] + 1;
for u=1:U
    Xd_non(u,ind(u)) = 1;
end
Xd1 = [X1,d1];
Xd2 = [X2,d2];
Xd = [-Xd_non;Xd1;Xd2];
A = Xd;
AtopA = Xd * Xd';
end