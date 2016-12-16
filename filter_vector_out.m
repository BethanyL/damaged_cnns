function [matrices_as_vector, damaged_number] = filter_vector_out(matrices_as_vector, ...
    percentile_window, damage_amt, net_size)   
%
% called by filter_network to damage weights in decreasing order of magnitude
%
% INPUTS:
% matrices_as_vector
%       vector of weights to damage
% 
% percentile_window
%       scalar in [0,50]: filter from both left and right ends of histogram, 
%       so the window is actually 2*percentile_window in size.
%
% damage_amt
%       scalar, set the damaged weights to damage_amt. For now, we just use 0 here 
%       (blockage)
%
% net_size
%       scalar, number of weights (not including any eliminated by sparsification)
%
% OUTPUTS: 
% matrices_as_vector
%       same size as input matrices_as_vector, but weights have been damaged
%
% damaged_number
%       scalar, number of weights damaged 
%
    
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

upper_perc = prctile(values_to_check, 100 - percentile_window);
lower_perc = prctile(values_to_check, 0 + percentile_window);
damaged_number = 0;
for i = 1:length(matrices_as_vector)
    if (matrices_as_vector(i) > upper_perc || matrices_as_vector(i) < lower_perc)
        matrices_as_vector(i) = damage_amt;
        damaged_number = damaged_number + 1;
    end
end