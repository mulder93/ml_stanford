function [J, grad] = lrCostFunction(theta, X, y, lambda)
	% Initialize some useful values
	m = length(y); % number of training examples

	% You need to return the following variables correctly 
	h = sigmoid(X * theta);
    J = (-y' * log(h) - (1 - y)' * log(1 - h)) / m + lambda * sum((theta .^ 2)(2:size(theta))) / (2 * m);
    grad = X' * (h - y) / m + lambda * [0;theta(2:size(theta))] / m;
end
