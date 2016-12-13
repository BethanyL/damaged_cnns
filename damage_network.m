function [damaged_matrices, num_damaged, damage_indices] = damage_network(network_matrices, ...
    dmg_size, pie_chart, coeff, sigma, net_size, high_weight)
    
tstart = tic; 
matrix_shapes = get_matrix_shapes(network_matrices);
matrices_as_vector = vectorize_network(network_matrices);

damage_indices = get_damage_indices(matrices_as_vector, dmg_size, net_size);
[matrices_as_vector(damage_indices), num_damaged] = damagefn(matrices_as_vector(damage_indices), ...
    pie_chart, high_weight, coeff, sigma);

damaged_matrices = reshape_matrices(matrices_as_vector, matrix_shapes);
num_damaged = length(damage_indices); % includes transmission 


end

