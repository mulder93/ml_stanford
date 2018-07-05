function label = predictOneVsAll(all_theta, X)
	examplesCount = size(X, 1);

	% Add ones to the X data matrix
	X = [ones(examplesCount, 1) X];

	label = zeros(size(X, 1), 1);
	[probability, label] = max(sigmoid(X * all_theta'), [], 2);
end
