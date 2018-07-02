function [cost, grad] = costFunctionReg(theta, X, y, lambda)
    examplesCount = length(y);
    hypothesis = sigmoid(X * theta);
    thetaZeroFirst = [0;theta(2:size(theta))];

    cost = (-y' * log(hypothesis) - (1 - y)' * log(1 - hypothesis)) / examplesCount;
    cost = cost + lambda * thetaZeroFirst' * thetaZeroFirst / (2 * examplesCount); %regularization

    grad = X' * (hypothesis - y) / examplesCount;
    grad = grad + lambda * thetaZeroFirst / examplesCount; %regularization
end
