function result = predict(Theta1, Theta2, X)
	result = zeros(size(X, 1), 1);
	X = [ones(size(X, 1), 1), X];

	layer2 = sigmoid(X * Theta1');
	layer2 = [ones(size(layer2, 1), 1), layer2];

	hypothesis = sigmoid(layer2 * Theta2');
	[probatility, result] = max(hypothesis, [], 2);
end
