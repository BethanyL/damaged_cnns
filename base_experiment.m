function base_experiment(expnum, pie_chart, damages_values, detailed_file_flag, ...
    max_trials, histogram_flag, filter_type, aging_flag, header_string, ...
    coeff, sigma, net_file_name, label_to_truth, conv_layers, sparsity_cutoff, ...
    batch_list, actual_test_image_labels)


if strfind(expnum,'face')
    face_flag = 1;
else 
    face_flag = 0;
end

filedir = sprintf('./exp%s/',expnum);
mkdir(filedir)

damages_values

net = setup_experiment(net_file_name, face_flag);

matrices_to_damage = get_conv_layers(net, conv_layers, face_flag);
matrices_as_vector = vectorize_network(matrices_to_damage);
high_weight = prctile(abs(matrices_as_vector(:)),95);

net_size = length(matrices_as_vector);
clear matrices_as_vector

if sparsity_cutoff
    [matrices_to_damage, num_removed] = sparsify_network(matrices_to_damage, sparsity_cutoff, net_size);
    net_size = net_size - num_removed;
end
    
accuracies = zeros(length(damages_values), 3);
trial_counter = 1;

% Damage and file output loop:
while 1
    fprintf('starting while loop (new trial)\n');
    file_name = initialize_new_file(header_string, trial_counter, expnum);
    dmg_counter = 1;
    matrices_to_damage_this_trial = matrices_to_damage;
    
    for dmg_size = damages_values
        fprintf('starting dmg_size %1.3f\n',dmg_size);
        if histogram_flag
            default_damage_amount = 0;
            [damaged_matrices, num_damaged] = filter_network(...
                matrices_to_damage_this_trial, dmg_size, ...
                default_damage_amount, filter_type, net_size);
        else
            [damaged_matrices, num_damaged] = damage_network(matrices_to_damage_this_trial, dmg_size, pie_chart, coeff, sigma, net_size, high_weight);
        end

        predicted_vectors = get_output_class_vectors(net, damaged_matrices, conv_layers, batch_list, face_flag); 
        predicted_test_image_labels = get_predicted_labels(predicted_vectors, label_to_truth, face_flag);
        network_accuracy = get_network_accuracy(actual_test_image_labels, predicted_test_image_labels, face_flag);
        
        if detailed_file_flag
            output_data_to_csv(file_name, dmg_size, num_damaged, trial_counter, ...
                actual_test_image_labels, predicted_test_image_labels, ...
                predicted_vectors, face_flag);
        end
        
        if aging_flag
            matrices_to_damage_this_trial = damaged_matrices;
        end
        
        accuracies(dmg_counter, 1) = dmg_size;
        accuracies(dmg_counter, 2) = num_damaged;
        accuracies(dmg_counter, 3) = network_accuracy;
        dmg_counter = dmg_counter + 1;
    end
    if ~detailed_file_flag
        output_summary_data_to_csv(file_name, accuracies, trial_counter);
    end
    fprintf('Trials completed: %d\n', trial_counter);
    trial_counter = trial_counter + 1;
    
    if trial_counter > max_trials
        break
    end
    
end