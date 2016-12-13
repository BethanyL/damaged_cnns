expnum = '1_peppers';

load('base_params.mat');
damages_values = 0:.1:.99;
detailed_file_flag = 1;
max_trials = 1;


actual_test_image_labels = 735;
% in network, class 946: bell pepper
% label_to_truth(946) = 735

batch_list = {'peppers.mat'};


base_experiment(expnum, pie_chart, damages_values, detailed_file_flag, ...
    max_trials, histogram_flag, filter_type, aging_flag, header_string, ...
    coeff, sigma, net_file_name, label_to_truth, conv_layers, sparsity_cutoff, ...
    batch_list, actual_test_image_labels);