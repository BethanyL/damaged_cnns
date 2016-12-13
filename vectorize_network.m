function vector = vectorize_network(network_matrices)

sizes = zeros(length(network_matrices), 1);
for j = 1:length(sizes)
    sizes(j) = numel(network_matrices{j});
end

vector = zeros(sum(sizes), 1);

cur = 1;
for j = 1:length(network_matrices)
    vector(cur:cur + sizes(j) - 1) = reshape(network_matrices{j}, ...
        sizes(j), 1);
    cur = cur + sizes(j);
end


