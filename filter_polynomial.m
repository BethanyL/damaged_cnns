function y = filter_polynomial(x, coeff, sigma)
    
y = coeff(1) * x.^2 + coeff(2) * x + coeff(3) + sigma * randn(length(x),1);


end

