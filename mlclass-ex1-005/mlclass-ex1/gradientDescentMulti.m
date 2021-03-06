function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

thetaLength = length(theta);
newTheta = zeros(thetaLength, 1);
for iter = 1:num_iters

    fprintf(' theta = [%.0f %.0f %.0f]\n', [theta(:)]');
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
	for index = 1:thetaLength
		newTheta(index) = theta(index) - alpha * sum((X * theta - y).* X(:,index))/ m;
	end

	theta = newTheta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end

% X = [2104 3;1600 3;2400 3;1416 2;3000 4;1985 4;1534 3;1427 3;1380 3;1494 3]
% y = [399900;329900;369000;232000;539900;299900;314900;198999;212000;242500]