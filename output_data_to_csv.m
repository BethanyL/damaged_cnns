% handles printing everything to .csv file. 
function output_data_to_csv(file_name, damage_size, num_damaged, trial_number, actual_labels, predicted_labels, class_scores, face_flag)

fd = fopen(file_name, 'a'); 
if face_flag
    % class_scores is num_images x num_features 
    num_images = size(class_scores, 1); 
    for j = 1:num_images
        fprintf(fd, '%d,%f,%f,%f,%d,%d,%d,', j, damage_size, num_damaged, trial_number);
        for jj = 1:num_images
            distance = norm(class_scores(j,:) - class_scores(jj,:));
            fprintf(fd, '%f,', distance);
        end
        fprintf(fd,'\n');
    end
            
else
    indices = range(length(actual_labels));

    for i = 1:length(actual_labels)
        if actual_labels(i) - predicted_labels(i) == 0
            is_wrong = 0;
        else
            is_wrong = 1;
        end
        fprintf(fd, '%d,%f,%f,%f,%d,%d,%d,', i, damage_size, num_damaged, trial_number, actual_labels(i), predicted_labels(i), is_wrong);
        for j = 1:size(class_scores,2)
            % loop over classes 
            fprintf(fd, '%f,', class_scores(i,j));
        end
        fprintf(fd,'\n');
    end
end
fclose(fd);