function lengths = get_vector_lengths(matrix_shapes)
% input is the shape of each weight matrix 
% returns vector of how many elements are in each weight matrix 

num_shapes = size(matrix_shapes, 1);
lengths = ones(num_shapes,1);
for j = 1:num_shapes
    lengths(j) = prod(matrix_shapes(j,:));
end