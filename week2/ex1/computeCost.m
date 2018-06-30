function cost = computeCost(X, y, theta)
	examplesCount = length(y);
	currentError = hypothesisError(X, y, theta);
	cost = currentError' * currentError / (2 * examplesCount);
end
