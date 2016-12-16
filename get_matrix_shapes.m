function list_of_shapes = get_matrix_shapes(network_matrices)
% returns shapes of original network matrices so that after we flatten the matrices we can 
% return to the previous shape

    list_of_shapes = zeros(length(network_matrices), length(size(network_matrices{1})));
    for j = 1:length(network_matrices)
        list_of_shapes(j, :) = size(network_matrices{j});
    end
end

