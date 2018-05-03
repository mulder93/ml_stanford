function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for i = 1:m
    z2 = Theta1 * [1 X(i,:)]';
    a2 = sigmoid(z2);
    z3 = Theta2 * [1; a2];
    h = sigmoid(z3);

    yi = (1:num_labels == y(i))';
    delta3 = h - yi;
    delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2);

    big_delta_2 = delta3 * [1; a2]';
    big_delta_1 = delta2 * [1 X(i,:)];

    Theta2_grad += big_delta_2 / m;
    Theta1_grad += big_delta_1 / m;

    J += (yi' * log(h) + (1 - yi') * log(1 - h)) / (-m);
endfor

% Regularization
J += (sum(sum(Theta1(:,[2:end]) .^ 2)) + sum(sum(Theta2(:,[2:end]) .^ 2))) * (lambda / (2 * m));
Theta2_grad(:,2:end) += lambda * Theta2(:,2:end) / m;
Theta1_grad(:,2:end) += lambda * Theta1(:,2:end) / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
