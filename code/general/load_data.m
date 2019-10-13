function [data, y_flip] = load_data(dataset_name)
	if strcmp(dataset_name, 'simulation') == 1
	    data = load('data/simulation/data.mat');
	    data = data.data;
	    y_flip = data(:,5);
	elseif strcmp(dataset_name, 'age') == 1
	    data = load('data/age/agedataset.mat');
	    data = data.age;
	    data_flip = load('data/age/splits_age_user.mat');
	    y_flip = data_flip.data(:,3);
	elseif strcmp(dataset_name, 'college') == 1
	    data = load('data/college/college.mat');
	    data = data.pair;
	    data_flip = load('data/college/splits_college_user.mat');
	    y_flip = data_flip.data(:,3);
	else
	    disp('Data do not exist!');
	end
end