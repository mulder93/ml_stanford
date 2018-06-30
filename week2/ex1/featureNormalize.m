function [normalizedX, mu, sigma] = featureNormalize(X)
	examplesCount = length(X);
	mu = mean(X, 1); ;
	sigma = std(X, 0, 1);
	normalizedX = (X - repmat(mu, examplesCount, 1)) ./ repmat(sigma, examplesCount, 1);
end
