function [damaged_matrices, num_damaged] = filter_network(network_matrices, ...
    percentile_window, damage_amt, filter_type, net_size)
%
% Damage weights in order of magnitude ("filter" the histogram of weights)
% 
% INPUTS:
%
% network_matrices
%       matrices of weights that we're going to damage 
%
% percentile_window
%       scalar in [0,50]: filter from median + and - the percentile window size, 
%       so the window is actually 2*percentile_window in size.
%
% damage_amt
%       scalar, set the damaged weights to damage_amt. For now, we just use 0 here 
%       (blockage)
%
% filter_type
%       string, Can be "inside" (for damaging the weights in increasing order of 
%       magnitude) or "outside" (for damaging the weights in decreasing order)
% 
% net_size
%       scalar, number of weights (not including any eliminated by sparsification)
%
% OUTPUTS:
%
% damaged_matrices
%       same size as network_matrices, but weights have been damaged
%
% num_damaged
%       scalar, number of weights damaged 
%

    matrix_shapes = get_matrix_shapes(network_matrices);
    matrices_as_vector = vectorize_network(network_matrices);
    if strcmp(filter_type,'inside')
        [return_vector, num_damaged] = filter_vector_in(matrices_as_vector, ...
            percentile_window, damage_amt, net_size);
    elseif strcmp(filter_type, 'outside')
        [return_vector, num_damaged] = filter_vector_out(matrices_as_vector, ...
            percentile_window, damage_amt, net_size);
    end
    
    return_vector = single(return_vector);
    damaged_matrices = reshape_matrices(return_vector, matrix_shapes);


end

