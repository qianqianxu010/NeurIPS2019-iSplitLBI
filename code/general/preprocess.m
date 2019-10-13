%% preprocess: remove number of user that less than 5
function [data_split, user_id_new, n_user_new, n_item_new] = preprocess(data, y_flip)
	user = data(:,1);
	user_id = unique(user);
	n_user = length(user_id);
	user_vec = zeros(n_user,1);
	for u = 1:n_user
	    user_vec(u) = sum(user == user_id(u));
	end
	user_id_new = user_id(user_vec > 6);
	data = data(ismember(user,user_id_new),:);
	y_flip = y_flip(ismember(user,user_id_new));
	comparison = data(:,2:3);
	item_id = unique([comparison(:,1);comparison(:,2)]);
	n_item_new = length(item_id);
	y = data(:,4);
	user = data(:,1);
	n_user_new = length(user_id_new);

	data_split.comp = comparison;
	data_split.y = y;
	data_split.y_flip = y_flip;
	data_split.user = user;
end