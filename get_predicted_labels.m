% returns labels predicted by the network
function labels = get_predicted_labels(predicted_vectors, label_to_truth, face_flag)

if face_flag
    % predicted_vectors is num_images x num_features, batch-by-batch
    % label_to_truth here is threshold for same person / not same person 
    tau = label_to_truth; 
    
    num_images = size(predicted_vectors, 1);
    num_batches = 5; % should do programatically 
    num_people = num_images/num_batches;
    
    labels = zeros(num_images, 2*(num_batches-1));  
    image_counter = 1;
    for k = 1:num_batches
        for j = 1:num_people
            % compare each image to the other ones of the same person
            for kk = k+1:num_batches
                if norm(predicted_vectors(image_counter, :) - predicted_vectors(image_counter + num_people*(kk-k), :)) >= tau
                    % think they're different people
                    labels(image_counter, kk-1) = 1;
                else
                    % think they're the same people 
                    labels(image_counter, kk-1) = -1;
                end
            end
            % and then 4 that are different 
            for kk = 1:2
                other_image = mod(image_counter + kk, num_images); % want 1 -> num_images 
                if other_image == 0
                    other_image = num_images;
                end
                if norm(predicted_vectors(image_counter, :) - predicted_vectors(other_image, :)) >= tau
                    % think they're different people
                    labels(image_counter, 4+kk) = 1;
                else
                    % think they're the same 
                    labels(image_counter, 4+kk) = -1;
                end
            end
                
            image_counter = image_counter + 1; 
        end
    end
else
    % imageNet
    [~, best] = max(predicted_vectors, [], 2);

    labels = zeros(length(best),1);
    for j = 1:length(best)
        % translate FROM network output TO answers from dataset
        labels(j) = label_to_truth(best(j));
    end
end
    
