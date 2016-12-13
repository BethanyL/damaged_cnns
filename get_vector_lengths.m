function lengths = get_vector_lengths(matrix_shapes)

num_shapes = size(matrix_shapes, 1);
lengths = ones(num_shapes,1);
for j = 1:num_shapes
    lengths(j) = prod(matrix_shapes(j,:));
end