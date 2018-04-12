function [J, grad] = costFunction(theta, X, y)
    % Initialize some useful values
    m = length(y); % number of training examples

    h = sigmoid(X * theta);
    J = (-y' * log(h) - (1 - y)' * log(1 - h)) / m;
    grad = X' * (h - y) / m;
end
