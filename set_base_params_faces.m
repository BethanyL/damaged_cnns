function set_base_params_faces()

load('base_params.mat')

net_file_name = 'vgg_face';

conv_layers = [1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29, 32, 35];

number_batches = 5; 
batch_list = cell(number_batches, 1);
load('ourSetPeople')
actual_test_image_labels = repmat(ourSetPeople, 1, number_batches);
count = 1;
for j = 1:number_batches
    batch_list{j} = sprintf('batch%d_faces.mat',j);
end

label_to_truth = 1.2; 


save('base_params_faces.mat')



