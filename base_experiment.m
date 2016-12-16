function base_experiment(expnum, pie_chart, damages_values, detailed_file_flag, ...
    max_trials, histogram_flag, filter_type, aging_flag, header_string, ...
    coeff, sigma, net_file_name, label_to_truth, conv_layers, sparsity_cutoff, ...
    batch_list, actual_test_image_labels)

% base experiment: lots of parameters so that many things can be varied
% NOTE: will run until with random trials until you kill it or you reach
%       max_trials
%
% INPUTS:
%
% expnum
%           a string giving the experiment "number," such as "2_faces" - this 
%           determines the subdirectory where the results are saved. Use "face"
%           in this string if you are using the VGG-face network, so that 
%           the code can specialize for that network.
%
% pie_chart
%           vector of length 4, each in [0,1]: distribution of types of damage 
%           (blockage, reflection, filtering, transmission), adds up to 1
%
% damages_values
%           For random damage: vector of values p, each in [0,1]: percentage of 
%           weights to damage. 
%           For damage in order of magnitude: vector of values in [0, 50]: percentage
%           of weights to damage on either side of the histogram (eliminate window 
%           of histogram of weights that is of width 2*p)
%           We loop over this vector 
%           
% detailed_file_flag
%           if non-zero, save detailed csv file, with a row for each example in the
%           test set. Otherwise, save summary csv file with an accuracy for each 
%           damage value
%
% max_trials
%           integer, maximum number of random trials to run. (This counts the 
%           outer loop, which contains an inner loop over the entries in 
%           damages_values.)
%
% histogram_flag
%           if non-zero, damage in order of magnitude. Otherwise, damage randomly.
%       
% filter_type
%           Used if histogram_flag is non-zero. Can be "inside" (for damaging 
%           the weights in increasing order of magnitude) or "outside" (for 
%           damaging the weights in decreasing order)
%
% aging_flag
%           If non-zero, accumulate damage as we loop over damages_values, instead
%           of starting over with a healthy network again (TBI).
%
% header_string
%           String to print at the top of the csv file (i.e. headings for columns)
%
% coeff
%           Vector of 3 coefficients for the low-pass filter applied to weights. 
%           Is used by filter_polynomial.m 
%
% sigma
%           scalar: standard deviation of randomness added to low-pass filter
%           in filter_polynomial.m 
%
% net_file_name
%           string for name of file containing the CNN. Used by setup_network.m. 
%
% label_to_truth
%           for ImageNET: vector that translates from the class ordering of the CNN
%           to the class ordering of the dataset. 
%           for VGG-face: scalar threshold for deciding if two images are the same
%           person or not
%
% conv_layers
%           vector list of which layers in the CNN are convolutional (contain 
%           weights we want to damage)
%
% sparsity_cutoff
%           scalar in [0,1] determining the percentage of weights to remove 
%           from the network at the beginning (to sparsify it). Will remove
%           weights in increasing order of magnitude.
%
% batch_list
%           list of strings: filenames of batches of images that should be used
%           as test set
%
% actual_test_image_labels
%           vector of true class labels. All of the image labels for all batches
%           are included in one vector.
%


% If used string "face" in experiment number, know that we're using the VGG-face
% network. Unfortunately that means special cases in the code.
if strfind(expnum,'face')
    face_flag = 1;
else 
    face_flag = 0;
end

filedir = sprintf('./exp%s/',expnum);
mkdir(filedir)

damages_values

net = setup_experiment(net_file_name, face_flag);

% get all the weights in the network and flatten them into a vector
matrices_to_damage = get_conv_layers(net, conv_layers, face_flag);
matrices_as_vector = vectorize_network(matrices_to_damage);

% we use this for the low-pass filter when damaging to temporarily rescale the weights 
high_weight = prctile(abs(matrices_as_vector(:)),95);

net_size = length(matrices_as_vector);
clear matrices_as_vector

if sparsity_cutoff
    % we want to sparsify the network before we use it: eliminate the smallest
    % P percentage of weights (in magnitude), where P = sparsity_cutoff
    [matrices_to_damage, num_removed] = sparsify_network(matrices_to_damage, sparsity_cutoff, net_size);
    net_size = net_size - num_removed;
end
    
accuracies = zeros(length(damages_values), 3);
trial_counter = 1;

% loop through random trials
% runs until killed or up to max_trials times
while 1
    fprintf('starting while loop (new trial)\n');
    file_name = initialize_new_file(header_string, trial_counter, expnum);
    dmg_counter = 1;
    matrices_to_damage_this_trial = matrices_to_damage;
    
    % loop over different amounts of damage 
    for dmg_size = damages_values
        fprintf('starting dmg_size %1.3f\n',dmg_size);
        if histogram_flag
            % here, damage weights in order of magnitude
            default_damage_amount = 0;
            [damaged_matrices, num_damaged] = filter_network(...
                matrices_to_damage_this_trial, dmg_size, ...
                default_damage_amount, filter_type, net_size);
        else
            % here, damage weights randomly 
            [damaged_matrices, num_damaged] = damage_network(matrices_to_damage_this_trial, dmg_size, pie_chart, coeff, sigma, net_size, high_weight);
        end

        % get class labels for each image
        predicted_vectors = get_output_class_vectors(net, damaged_matrices, conv_layers, batch_list, face_flag); 
        predicted_test_image_labels = get_predicted_labels(predicted_vectors, label_to_truth, face_flag);
        
        % compare to true class labels and get the accuracy 
        network_accuracy = get_network_accuracy(actual_test_image_labels, predicted_test_image_labels, face_flag);
        
        if detailed_file_flag
            output_data_to_csv(file_name, dmg_size, num_damaged, trial_counter, ...
                actual_test_image_labels, predicted_test_image_labels, ...
                predicted_vectors, face_flag);
        end
        
        if aging_flag
            % if aging (instead of TBI), keep accumulating damage on the
            % same network instead of starting over with a new healthy one 
            matrices_to_damage_this_trial = damaged_matrices;
        end
        
        accuracies(dmg_counter, 1) = dmg_size; % percentage of weights damaged
        accuracies(dmg_counter, 2) = num_damaged; % number of weights damaged
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