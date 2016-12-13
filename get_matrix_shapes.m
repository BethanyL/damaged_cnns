% returns shapes of original network matrices for reshaping
function list_of_shapes = get_matrix_shapes(network_matrices)
    list_of_shapes = zeros(length(network_matrices), length(size(network_matrices{1})));
    for j = 1:length(network_matrices)
        list_of_shapes(j, :) = size(network_matrices{j});
    end
end

