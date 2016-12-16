function [damaged_matrices, num_damaged, damage_indices] = damage_network(network_matrices, dmg_size, pie_chart, coeff, sigma, net_size, high_weight)

% randomly damages weights
%
% INPUTS:
% network_matrices
%           matrices of weights that we're going to damage 
%
% dmg_size
%           scalar in [0,1], percentage of weights to damage     
%
% pie_chart
%           vector of length 4, each in [0,1]: distribution of types of damage 
%           (blockage, reflection, filtering, transmission), adds up to 1
%
% coeff
%           Vector of 3 coefficients for the low-pass filter applied to weights. 
%           Is used by filter_polynomial.m 
%
% sigma
%           scalar: standard deviation of randomness added to low-pass filter
%           in filter_polynomial.m 
%
% net_size
%           scalar, number of weights (not including any eliminated by 
%           sparsification)
%
% high_weight
%           scalar, magnitude of weight in 95th percentile of the weight magnitudes
%           (used for filtering in weight_filter.m to temporarily rescale the weights)
%
% OUTPUTS:
% damaged_matrices
%           same size as network_matrices, but weights have been damaged
%
% num_damaged
%           scalar, number of weights damaged 
%
% damage_indices
%           indices of weights that were damaged
%

% get the sizes of the weight matrices so can return to that shape later
% then flatten into one vector
matrix_shapes = get_matrix_shapes(network_matrices);
matrices_as_vector = vectorize_network(network_matrices);

% decide which indices we'll damage, then apply the damage
damage_indices = get_damage_indices(matrices_as_vector, dmg_size, net_size);
[matrices_as_vector(damage_indices), num_damaged] = damagefn(matrices_as_vector(damage_indices), ...
    pie_chart, high_weight, coeff, sigma);

% go back from vector shape to original shape 
damaged_matrices = reshape_matrices(matrices_as_vector, matrix_shapes);
num_damaged = length(damage_indices); % includes transmission 


end

