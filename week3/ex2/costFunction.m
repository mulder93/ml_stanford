function [cost, grad] = costFunction(theta, X, y)
    examplesCount = length(y);
    hypothesis = sigmoid(X * theta);

    cost = (-y' * log(hypothesis) - (1 - y)' * log(1 - hypothesis)) / examplesCount;
    grad = X' * (hypothesis - y) / examplesCount;
end
