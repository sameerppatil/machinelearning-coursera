function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


htheta = X * theta;  % X is mx1, theta is 1x1

predicterror = (htheta - y) .^ 2;

theta_j = [ 0; theta(2:end)];

J = (1/(2*m))*sum(predicterror) + (lambda/(2*m))*sum(theta_j .^ 2);

grad = (1/m) .* (X' * (htheta - y)) + (lambda/m) * theta_j;


% =========================================================================

grad = grad(:);

end
