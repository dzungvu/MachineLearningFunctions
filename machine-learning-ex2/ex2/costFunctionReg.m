function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

z = X * theta;
hx = sigmoid(z); %hypothesis
y1 = -1 * y .* log(hx); %first part of cost function

oneMatrix = ones(m, 1);
y2 = (oneMatrix - y) .* log(oneMatrix - hx); %second part of cost function

%Regularization will only be apply when j > 0. When j = 0, don't add to sum
reg_theta = theta(2:n, :)
reg = (lambda / (2 * m)) * sum((reg_theta .^ 2))
%Cost function
J = (1 / m) * sum((y1 - y2)) + reg

%Init grad with formular of j = 0, after that, modify for j > 0 (mean theta > 1)
grad = (1 / m) * X' * (hx - y)

%Reg for j > 0 (theta > 1)
reg_grad = (lambda / m) * reg_theta
%Re-Compute for grad(2:length(grad))
grad(2:length(grad)) = grad(2:length(grad)) + reg_grad;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
