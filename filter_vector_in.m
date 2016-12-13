% helper function for filter, inside damage
function [matrices_as_vector, damaged_number] = filter_vector_in(matrices_as_vector, ...
    percentile_window, damage_amt, net_size)
    
vec_size = length(matrices_as_vector);
if vec_size > net_size
    % means that we already set some thresholded section to 0. Don't want to count those zeros
    % in calculating percentile
    num_thresholded = vec_size - net_size;
    [~,sorted_ind] = sort(matrices_as_vector); %indices that would sort matrices_as_vector
    sorted_ind = sorted_ind(num_thresholded+1:vec_size); % cut off num_thresholded from front
    values_to_check = matrices_as_vector(sorted_ind);
else
    values_to_check = matrices_as_vector;
end
upper_perc = prctile(values_to_check, 50 + percentile_window);
lower_perc = prctile(values_to_check, 50 - percentile_window);
damaged_number = 0;
for i = 1:length(matrices_as_vector)
    if (matrices_as_vector(i) <= upper_perc && matrices_as_vector(i) >= lower_perc)
        matrices_as_vector(i) = damage_amt;
        damaged_number = damaged_number + 1;
    end
end