function cost = computeCostMulti(X, y, theta)
	currentError = hypothesisError(X, y, theta);
	examplesCount = length(y);
	cost = currentError' * currentError / (2 * examplesCount);
end
