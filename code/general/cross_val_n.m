%% cross_val_n: cross-validation for n times
function [mxcro_outlier_id, mxcro_dense_vec, ps_dense_t, cs_t, ps_supp] = cross_val_n(data, n_repeat, user_id, n_user, n_item, cv_param, splitlbi_param, use_micro)
	mxcro_dense_vec = zeros(n_repeat,1);
	mxcro_outlier_id = cell(n_repeat);

	ps_dense_t = cell(n_repeat);
	cs_t = cell(n_repeat);
	ps_supp = cell(n_repeat);

	for i=1:n_repeat
	    % cross-validation setting (specific)
	    cv_param.seed_test = i;
	    cv_param.seed_cross = i;
	    
	    % Implement Split LBI
	    result = Split_tie_grouplasso(data,splitlbi_param,cv_param,use_micro);
	    
	    mxcro_outlier_id{i} = get_outlier_id(result.gamma, user_id, n_user, n_item);
	    mxcro_dense_vec(i) = result.test_mxcro_dense;
	    
	    % c,p result
	    ps_dense_t{i} = result.ps_dense_t;
	    cs_t{i} = result.cs_t;
	    ps_supp{i} = result.ps_supp;
	end
end