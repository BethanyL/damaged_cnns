% returns final output values for every class by image.
% output score predicted_vectors, which is then givent to get_predicted_labels
function scores = get_output_class_vectors(net_to_damage, damaged_matrices, conv_layers, batch_list, face_flag)
% here, net_to_damage should be of class convNet

for j = 1:length(conv_layers)
    if face_flag
        % ending layers should be already gone from setup_network
        net_to_damage.net.layers{conv_layers(j)}.weights{1} = damaged_matrices{j};
    else
        net_to_damage.layers{conv_layers(j)}.weights{1} = damaged_matrices{j};
    end
end
clear damaged_matrices

if face_flag
    config.paths.face_model_path = 'face_model.mat';
    faceDet = lib.face_detector.dpmCascadeDetector(config.paths.face_model_path);
end

num_batches = length(batch_list);

scores_cell = cell(num_batches,1);
sizes = zeros(num_batches, 1); % number of images in each batch 
for j = 1:num_batches
    if face_flag
        % returns num_images x num_features 
        % here, net_to_damage should be of class convNet
        scores_cell{j} = get_facial_features(faceDet, net_to_damage, batch_list{j});
    else
        % returns num_images x num_classes 
        scores_cell{j} = classify_image_imagenet(net_to_damage, batch_list{j});
    end
    sizes(j) = size(scores_cell{j},1);
end

% flatten scores, now that we know how many images are in each batch 
scores = zeros(sum(sizes), size(scores_cell{1},2));
count = 0;
for j = 1:num_batches
    scores(count + 1: count + sizes(j), :) = scores_cell{j};
    count = count + sizes(j); 
end