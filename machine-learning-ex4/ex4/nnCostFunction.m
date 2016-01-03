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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% convert y to 1s and 0s
Y = zeros(m, num_labels);
for c = 1:num_labels
    Y(:,c) = (y == c);
end

X = [ones(m,1) X];
a1 = X;
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3); % a3 = h(x)

lambdaVector1 = ones(size(Theta1,2),1).*lambda;
lambdaVector1(1,1) = 0;

lambdaVector2 = ones(size(Theta2,2),1).*lambda;
lambdaVector2(1,1) = 0;

J = (sum( sum(-1.*Y.*log(a3),2) - sum((1 - Y).*log(1 - a3),2) ) / m ) + ...
    ( sum(Theta1.^2)*lambdaVector1 + sum(Theta2.^2)*lambdaVector2 )/(2*m);

delta1 = zeros(size(Theta1,1), size(Theta1,2));
delta2 = zeros(size(Theta2,1), size(Theta2,2));

for i = 1 : m
    % step 1
    backa1 = X(i,:);
    backz2 = backa1*Theta1';
    backa2 = sigmoid(backz2);
    backa2 = [1 backa2];
    backz3 = backa2*Theta2';
    backa3 = sigmoid(backz3);
    
    % step 2
    d3 = backa3 - Y(i,:);
    
    % step 3
    d2 = d3*Theta2(:,2:end).*sigmoidGradient(backz2);
    
    % step 4
    delta1 = delta1 + d2'*backa1;
    delta2 = delta2 + d3'*backa2;
end

Theta1_grad = delta1 / m;
Theta2_grad = delta2 / m;


Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1 = Theta1 * lambda / m;
Theta2 = Theta2 * lambda / m;

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;







% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
