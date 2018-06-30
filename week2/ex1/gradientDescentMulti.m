function [theta, costHistory] = gradientDescentMulti(X, y, theta, alpha, iterationsCount)
    examplesCount = length(y);
    costHistory = zeros(iterationsCount, 1);

    for iteration = 1:iterationsCount
        currentError = hypothesisError(X, y, theta);
        theta = theta - alpha * X' * currentError / examplesCount;
        costHistory(iteration) = computeCost(X, y, theta);
    end
end
