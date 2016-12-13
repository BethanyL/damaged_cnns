% filter network:
% filter_type = "inside", filters inside-out.
% filter_type = "outside", filters outside-in.
% filters from median + and - the percentile window size, so the window is actually 2*percentile_window in size.
function [damaged_matrices, num_damaged] = filter_network(network_matrices, ...
    percentile_window, damage_amt, filter_type, net_size)
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

