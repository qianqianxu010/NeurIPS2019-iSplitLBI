addpath('../split_lbi');

%% Load Dataset
dataset_name = 'age';  % 'age', 'college', 'simulation'
disp(['Load data of ' dataset_name '...']);
[data, y_flip] = load_data(dataset_name);

%% Remove number of user that less than 5
disp('Preprocess data...');
[data_split, user_id_new, n_user_new, n_item_new] = preprocess(data, y_flip);

% cross-validation setting (general)
n_repeat = 20;
cv_param.kfolds = 5;
cv_param.test_per = 0.2;
cv_param.class = 'BT';

% check if result folder exists
if ~exist('result', 'dir')
    mkdir('result')
end
dir_name = strcat('result/', dataset_name);
if ~exist(dir_name, 'dir')
    mkdir(dir_name)
end


%% Micro
% Split LBI setting
% = true: micro; = false: macro
use_micro = true;
lbi_param = get_lbi_param(dataset_name, use_micro);

% Run Split LBI
disp(['Start CV (micro) for ' num2str(n_repeat) ' times...']);
[micro_outlier_id, micro_dense_vec, ps_dense_t, cs_t, ps_supp] = cross_val_n(data_split, n_repeat, user_id_new, n_user_new, n_item_new, cv_param, lbi_param, use_micro);
disp(['Save result (micro)...']);
save(strcat(dir_name, '/micro_dense_vec.mat'),'micro_dense_vec');


%% Macro
% Split LBI setting
% = true: micro; = false: macro
use_micro = false;
lbi_param = get_lbi_param(dataset_name, use_micro);

% Run Split LBI
disp(['Start CV (macro) for ' num2str(n_repeat) ' times...']);
[macro_outlier_id, macro_dense_vec, ps_dense_t, cs_t, ps_supp] = cross_val_n(data_split, n_repeat, user_id_new, n_user_new, n_item_new, cv_param, lbi_param, use_micro);
disp(['Save result (macro)...']);
save(strcat(dir_name, '/macro_dense_vec.mat'),'macro_dense_vec');
