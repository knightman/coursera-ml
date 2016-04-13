function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
thLength = length(theta);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %theta_i = 1;    % create variable for theta index
    temp = theta;   % create temp var for theta values
    for i = 1:thLength    % for each index of theta

        temp(i,1) = sum((X * theta - y) .* X(:,i)); % create temp vector with sum of theta values, grad descent form
        disp(sprintf('Theta is '));
        disp(theta);
        disp(sprintf('Temp is '));
        disp(temp);
        disp(sprintf('Iter is '));
        disp(iter);

    end

    theta = theta - alpha * (1/m) * temp;   % update theta with temp values

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
