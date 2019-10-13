function [outlier_id_vec] = get_outlier_id(gamma, user_id, n_user, n_item)
	gamma_norm_vec = zeros(n_user,1);
	outlier_id_vec = [];
	for u = 1:n_user
	    gamma_norm_vec(u) = norm(gamma((u-1)*(n_item+1)+1:u*(n_item+1)));
	    if gamma_norm_vec(u) > 1e-6
	        outlier_id_vec = [outlier_id_vec;user_id(u)];
	    end
	end
end