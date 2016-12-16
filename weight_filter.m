function filtered_weights = weight_filter(weights_to_damage, high_weight, coeff, sigma)
%
% rescale weights, then apply low-pass filter and undo the rescaling 

if ~isempty(weights_to_damage)
    scaled_weights = weights_to_damage / high_weight; % mostly between -1 and 1
    signs = sign(scaled_weights);
    
    filtered_weights = signs .* filter_polynomial(abs(scaled_weights), ...
        coeff, sigma) * high_weight;

else
    % this if-else block might save some time in the common case of 
    % having empty weights vector 
    filtered_weights =  weights_to_damage;

end

