expnum = '1_faces_prez';
load('base_params_faces.mat');
detailed_file_flag = 1;
batch_list = cell(5,1);
for j = 1:5
    batch_list{j} = sprintf('batch_prez%d.mat',j);
end

base_experiment(expnum, pie_chart, damages_values, detailed_file_flag, ...
    max_trials, histogram_flag, filter_type, aging_flag, header_string, ...
    coeff, sigma, net_file_name, label_to_truth, conv_layers, sparsity_cutoff, ...
    batch_list, actual_test_image_labels);