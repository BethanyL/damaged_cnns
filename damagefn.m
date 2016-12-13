function [weights_to_damage, num_damaged] = damagefn(weights_to_damage, pie_chart, high_weight, coeff, sigma)

% Red (blockage), Orange (Reflection), Yellow (Filtering) and Green (Transmision)
num_weights = length(weights_to_damage);
num_types = nnz(pie_chart);
if num_types > 1
    % randomly split weights into four groups
    permuted_ind = randperm(num_weights);

    end_blocked = round(pie_chart(1)*num_weights);
    blocked_ind = permuted_ind(1:end_blocked-1);

    end_reflected = round(pie_chart(2)*num_weights) + end_blocked;
    reflected_ind = permuted_ind(end_blocked:end_reflected-1);

    end_filtered = round(pie_chart(3)*num_weights) + end_reflected;
    filtered_ind = permuted_ind(end_reflected:end_filtered-1);

    weights_to_damage(blocked_ind) = 0;
    weights_to_damage(reflected_ind) = .5 * weights_to_damage(reflected_ind);
    weights_to_damage(filtered_ind) = weight_filter(weights_to_damage(filtered_ind), high_weight, coeff, sigma);

    num_damaged = length(blocked_ind) + length(reflected_ind) + length(filtered_ind);
else
    % if they are all of same type, this saves a lot of time: randperm is
    % slow
    if pie_chart(1) == 1
        % blockage only: set all to 0
        num_damaged = length(weights_to_damage);
        weights_to_damage = 0;
    elseif pie_chart(2) == 1
        % reflection only: halve all weights
        weights_to_damage = .5 * weights_to_damage;
        num_damaged = length(weights_to_damage);
    elseif pie_chart(3) == 1
        % filtering only
        weights_to_damage = weight_filter(weights_to_damage, high_weight, coeff, sigma);
        num_damaged = length(weights_to_damage);
    end % final case is transmission only, so do no damage
end

        
    


end

