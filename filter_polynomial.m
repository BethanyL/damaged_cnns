function y = filter_polynomial(x, coeff, sigma)
%
% polynomial to do low-pass filter on weights
% add random noise with N(0, sigma) distribution
%
% INPUTS:
% x
%           vector of weights to damage, all temporarily rescaled
%           to be approximately in [0,1]
%           
% coeff
%           Vector of 3 coefficients for the polynomial
%
% sigma
%           scalar: standard deviation of randomness added
%
% OUTPUTS:
% y
%           same size as x, but with filter applied
%
    
y = coeff(1) * x.^2 + coeff(2) * x + coeff(3) + sigma * randn(length(x),1);


end

