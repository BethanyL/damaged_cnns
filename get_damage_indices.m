% returns random sample of indices to damage
function sample = get_damage_indices(matrices_as_vector, dmg_size, net_size)

% find returns indices of non-zero elements 
non_zero_elements = find(matrices_as_vector);

if dmg_size <= .6
    num_elements_to_damage = floor(dmg_size * net_size);
      
    % return a k-length list of unique elements chosen from population sequence
    fprintf('dmg_size is %f, num_elements_to_damage is %d out of %d options\n',dmg_size, ...
        num_elements_to_damage, length(non_zero_elements));
    linear_indices = randperm(length(non_zero_elements), ...
        min(num_elements_to_damage, length(non_zero_elements)));
else
    num_elements_to_skip = floor((1-dmg_size) * net_size);
    fprintf('dmg_size is %f, num_elements_to_skip is %d out of %d options\n',dmg_size, ...
        num_elements_to_skip, length(non_zero_elements));
    indices_to_skip = randperm(length(non_zero_elements), ...
        min(num_elements_to_skip, length(non_zero_elements)));
    linear_indices = setdiff(1:length(non_zero_elements), indices_to_skip);
end
% min here because could have not enough non-zero elements 

sample = non_zero_elements(linear_indices);
