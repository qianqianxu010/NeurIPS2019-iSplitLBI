function est_proj = proj_U(est_tmp,A,AtopA)

p = length(est_tmp);
cvx_precision best
           cvx_begin quiet
               variable s(p)
               minimize ( norm(s - est_tmp,2) );
               subject to:
                    A * s <= b;
               
           cvx_end
beta = s;
    
b = sparse(size(A,1),1);
b(U+1:end) = 1;

est_b = A * est_tmp - b;
lambda = AtopA \ est_b;
lambda(est_b < 1e6) = 0;
est_proj = est_tmp - A' * lambda;
end
