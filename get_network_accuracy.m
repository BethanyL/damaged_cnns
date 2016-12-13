% returns accuracy of the network
function score = get_network_accuracy(actual_labels, predicted_labels, face_flag)

if face_flag
    % first four columns are supposed to be same person (-1)
    FN = length(find(predicted_labels(:,1:4) == 1));
    FP = length(find(predicted_labels(:,5:8) == -1)); 
    num_errors = FN + FP; 
    score = 1 - num_errors/(length(predicted_labels)*4);

    % last four columns are supposed to be different people (+1)
else
    % imagenet
    errors = find(actual_labels(:) ~= predicted_labels(:));
    num_errors = length(errors);
    score = 1 - num_errors/length(actual_labels);
end