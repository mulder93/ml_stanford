%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

fprintf('Plot the convergence graph ...\n');

num_iters = 400;
start_theta = zeros(3, 1);
[theta, J1] = gradientDescentMulti(X, y, start_theta, 0.001, num_iters);
[theta, J2] = gradientDescentMulti(X, y, start_theta, 0.003, num_iters);
[theta, J3] = gradientDescentMulti(X, y, start_theta, 0.01, num_iters);
[theta, J4] = gradientDescentMulti(X, y, start_theta, 0.03, num_iters);
[theta, J5] = gradientDescentMulti(X, y, start_theta, 0.1, num_iters);
[theta, J6] = gradientDescentMulti(X, y, start_theta, 0.3, num_iters);
[theta, J7] = gradientDescentMulti(X, y, start_theta, 1, num_iters);

figure;
plot(1:50, J1(1:50), '-k;0.001;', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

hold on;
plot(1:50, J2(1:50), '-r;0.003;', 'LineWidth', 2);
plot(1:50, J3(1:50), '-g;0.01;', 'LineWidth', 2);
plot(1:50, J4(1:50), '-b;0.03;', 'LineWidth', 2);
plot(1:50, J5(1:50), '-y;0.1;', 'LineWidth', 2);
plot(1:50, J6(1:50), '-m;0.3;', 'LineWidth', 2);
plot(1:50, J7(1:50), '-c;1;', 'LineWidth', 2);
hold off;

fprintf('Running gradient descent ...\n');

alpha = 0.3; % seems good
[theta, J] = gradientDescentMulti(X, y, start_theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
values = ([1650 3] - mu) ./ sigma;
values = [1 values];
price = values * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
values = [1 1650 3];
price = values * theta;

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

