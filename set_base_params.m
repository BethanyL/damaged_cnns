function set_base_params()

pie_chart = [1, 0, 0, 0];
damages_values = 0:.01:1;
detailed_file_flag = 0;
max_trials = Inf;
histogram_flag = 0;
filter_type = 0;
aging_flag = 0;
header_string = 'image_index, damage_size, trial, correct_class, inferred_class, is_wrong, pred_0, pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8, pred_9\n';
coeff = [-.2774, .9094, -.0192];
sigma = .05;
net_file_name = 'imagenet-vgg-f.mat';
load('./label_to_truth.mat'); % load label_to_truth
conv_layers = [1, 5, 9, 11, 13, 16, 18, 20];
sparsity_cutoff = 0;

number_batches = 4; % 1a, 1b, 2a, 2b
batch_list = cell(number_batches, 1);
actual_test_image_labels = repmat(1:1000, 1, number_batches/2);
count = 1;
for j = 1:number_batches/2
    batch_list{count} = sprintf('batch%da.mat',j);
    batch_list{count+1} = sprintf('batch%db.mat',j);
    count = count + 2;
end


save('base_params.mat')



