% reshapes damaged vector into original network matrices
function matrices = reshape_matrices(matrix_as_vector, matrix_shapes)

% vector_lengths is a vector of 
vector_lengths = get_vector_lengths(matrix_shapes);
matrices = cell(size(vector_lengths, 1), 1); 

for j = 1:size(matrix_shapes,1)
    matrices{j} = single(reshape(...
            matrix_as_vector(sum(vector_lengths(1:(j-1)))+1:sum(vector_lengths(1:j))),...
                   matrix_shapes(j,:)));
end