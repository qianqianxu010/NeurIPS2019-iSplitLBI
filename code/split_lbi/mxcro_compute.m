function test_mxcro = mxcro_compute(index_test,y,y_flip,X,d1,d2,beta,delta,micro_ornot)
U = length(index_test);
p = size(X,2);

ind_flip = find(y ~= y_flip);
X(ind_flip,:) = -X(ind_flip,:);


d_unchange_ind = [1:p+1:U*(p+1)];
d_ind_all = [1:U*(p+1)];
d_change_ind = d_ind_all(~ismember(d_ind_all,d_unchange_ind));

d1(ind_flip,d_change_ind) = -d1(ind_flip,d_change_ind);
d2(ind_flip,d_change_ind) = -d2(ind_flip,d_change_ind);

X1 = [-ones(size(X,1),1),X];
X2 = [ones(size(X,1),1),X];

tp_n1 = 0;
fp_n1 = 0;
fn_n1 = 0;

tp_0 = 0;
fp_0 = 0;
fn_0 = 0;

tp_p1 = 0;
fp_p1 = 0;
fn_p1 = 0;

y_esti_1 = X1 * beta + d1 * delta;
y_esti_2 = X2 * beta + d2 * delta;
for u=1:U
    ind_u = index_test{u};
    y_test = y_flip(ind_u);
    y_est1 = y_esti_1(ind_u);
    y_est2 = y_esti_2(ind_u);
    
    
    tp_n1 = tp_n1 + sum(y_test == -1 & y_est2 < 0);
    fp_n1 = fp_n1 + sum(y_test ~= -1 & y_est2 < 0);
    fn_n1 = fn_n1 + sum(y_test == -1 & y_est2 >= 0);
    
    tp_0 = tp_0 + sum(y_test == 0 & (y_est2 >= 0 & y_est1 <= 0) );
    fp_0 = fp_0 + sum(y_test ~= 0 & (y_est2 >= 0 & y_est1 <= 0) );
    fn_0 = fn_0 + sum(y_test == 0 & (y_est2 < 0 | y_est1 > 0) );
    
    tp_p1 = tp_p1 + sum(y_test == 1 & y_est1 > 0);
    fp_p1 = fp_p1 + sum(y_test ~= 1 & y_est2 > 0);
    fn_p1 = fn_p1 + sum(y_test == 1 & y_est2 <= 0);
    
end

if micro_ornot == false
    pre_n1 = tp_n1 / (tp_n1 + fp_n1);
    pre_0 = tp_0 / (tp_0 + fp_0);
    pre_p1 = tp_p1 / (tp_p1 + fp_p1);
    if isnan(pre_n1)
        pre_n1 = 0;
    end
    if isnan(pre_0)
        pre_0 = 0;
    end
    if isnan(pre_p1)
        pre_p1 = 0;
    end
    precision = mean([pre_n1,pre_0,pre_p1]);
    
    rec_n1 = tp_n1 / (tp_n1 + fn_n1);
    rec_0 = tp_0 / (tp_0 + fn_0);
    rec_p1 = tp_p1 / (tp_p1 + fn_p1);
    if isnan(rec_n1)
        rec_n1 = 0;
    end
    if isnan(rec_0)
        rec_0 = 0;
    end
    if isnan(rec_p1)
        rec_p1 = 0;
    end
    recall = mean([rec_n1,rec_0,rec_p1]);
else
    precision = (tp_n1 + tp_0 + tp_p1) / (tp_n1 + tp_0 + tp_p1 + fp_n1 + fp_0 + fp_p1);
    recall = (tp_n1 + tp_0 + tp_p1) / (tp_n1 + tp_0 + tp_p1 + fn_n1 + fn_0 + fn_p1);
    if isnan(precision)
        precision = 0;
    end
    if isnan(recall)
        recall = 0;
    end
end
    
mxcro = 2 * precision * recall / (precision + recall);
if isnan(mxcro)
    mxcro = 0;
end
test_mxcro = mxcro;

% if micro_ornot == false
%     test_mxcro = [pre_n1,pre_0,pre_p1,rec_n1,rec_0,rec_p1];
% end
end
    
    
        