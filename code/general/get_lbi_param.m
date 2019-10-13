%% get_lbi_param: get parameters for iSplit LBI
function [lbi_param] = get_lbi_param(dataset_name, use_micro)
	lbi_param.fast_init = true;
	lbi_param.intercept = false;
	if use_micro
		if strcmp(dataset_name, 'simulation') == 1
		    lbi_param.nu = 10;
		    lbi_param.kappa = 10;
		    lbi_param.t_ratio = 50;
		    lbi_param.if_colledge = false;
		elseif strcmp(dataset_name, 'age') == 1
		    lbi_param.nu = 1;
		    lbi_param.kappa = 10;
		    lbi_param.t_ratio = 100;
		    lbi_param.if_colledge = true;
		elseif strcmp(dataset_name, 'college') == 1
		    lbi_param.nu = 1;
		    lbi_param.kappa = 10;
		    lbi_param.t_ratio = 100;
		    lbi_param.if_colledge = true;
		else
		    disp('Please customize the settings!');
		end
	else
		if strcmp(dataset_name, 'simulation') == 1
		    lbi_param.nu = 1;
		    lbi_param.kappa = 10;
		    lbi_param.t_ratio = 50;
		    lbi_param.if_colledge = false;
		elseif strcmp(dataset_name, 'age') == 1
		    lbi_param.nu = 1;
		    lbi_param.kappa = 10;
		    lbi_param.t_ratio = 200;
		    lbi_param.if_colledge = true;
		elseif strcmp(dataset_name, 'college') == 1
		    lbi_param.nu = 1;
		    lbi_param.kappa = 10;
		    lbi_param.t_ratio = 200;
		    lbi_param.if_colledge = true;
		else
		    disp('Please customize the settings!');
		end
	end
end