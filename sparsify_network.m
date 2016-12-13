function [sparsified_networks, num_removed] = sparsify_network(network_matrices, percentile_window, net_size)
    % sparsify network by removing small weights (within percentile_window)
    damage_amt = 0;
    filter_type = 'inside';
    [sparsified_networks, num_removed] = filter_network(network_matrices, ...
        percentile_window, damage_amt, filter_type, net_size);
    
end

