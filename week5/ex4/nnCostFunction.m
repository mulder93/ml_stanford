function [cost grad] = nnCostFunction(nnParams, ...
                                   inputLayerSize, ...
                                   hiddenLayerSize, ...
                                   labelsCount, ...
                                   X, y, lambda)
  % Reshape nnParams back into the parameters theta1 and theta2, the weight matrices
  % for our 2 layer neural network
  theta1Size = hiddenLayerSize * (inputLayerSize + 1);
  theta1 = reshape(nnParams(1:theta1Size), hiddenLayerSize, (inputLayerSize + 1));
  theta2 = reshape(nnParams((theta1Size + 1):end), labelsCount, (hiddenLayerSize + 1));

  examplesCount = size(X, 1);
  X = [ones(examplesCount, 1) X];
  cost = 0;
  theta1Grad = zeros(size(theta1));
  theta2Grad = zeros(size(theta2));

  for i = 1:examplesCount
      z2 = theta1 * X(i,:)';
      a2 = [1; sigmoid(z2)];
      z3 = theta2 * a2;
      hypothesis = sigmoid(z3);

      yi = 1:labelsCount == y(i);
      delta3 = hypothesis - yi';
      delta2 = (theta2' * delta3)(2:end) .* sigmoidGradient(z2);

      bigDelta2 = delta3 * a2';
      bitDelta1 = delta2 * X(i,:);

      theta2Grad += bigDelta2 / examplesCount;
      theta1Grad += bitDelta1 / examplesCount;

      cost += (yi * log(hypothesis) + (1 - yi) * log(1 - hypothesis)) / (-examplesCount);
  endfor

  % Regularization
  cost += (sum(sum(theta1(:,2:end) .^ 2)) + sum(sum(theta2(:,2:end) .^ 2))) * (lambda / (2 * examplesCount));
  theta2Grad(:,2:end) += lambda * theta2(:,2:end) / examplesCount;
  theta1Grad(:,2:end) += lambda * theta1(:,2:end) / examplesCount;

  % Unroll gradients
  grad = [theta1Grad(:) ; theta2Grad(:)];
end
