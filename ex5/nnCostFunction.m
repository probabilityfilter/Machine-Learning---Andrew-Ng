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

c=1:num_labels;
for i=1:m
Y(:,i) = c'==y(i);
a2 = sigmoid([1 X(i,:)]*Theta1');
h_theta = sigmoid([1 a2]*Theta2');
JJ(i) = (-1/m)*(log(h_theta)* Y(:,i) + (log(1-h_theta)*(1-Y(:,i))));
end;
J=sum(JJ)+(lambda/(2*m))*(sum(sum((Theta1(:,2:size(Theta1, 2)).^2))) + sum(sum((Theta2(:,2:size(Theta2, 2)).^2))));
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

for i=1:m
% Step 1
a1 = [1 X(i,:)]; %dim 1*401
z2 = a1*Theta1'; %dim 1*25
a2 = sigmoid(z2); %dim 1*25
a2 = [1 a2]; %dim 1*26
z3 = a2*Theta2'; %dim 1*10
a3 = sigmoid(z3); %dim 1*10

% Step 2
d3 = a3'-Y(:,i); %dim 10*1

% Step 3
d2_temp = Theta2' * d3; %dim 26*1
d2_temp = d2_temp(2:end); %dim 25*1
d2 = d2_temp'.*sigmoidGradient(z2); %dim 1*25

% Step 4 & 5
Theta1_grad = Theta1_grad + (d2'*a1)./m; %dim 25*401
Theta2_grad = Theta2_grad + (d3*a2)./m; %dim 10*26

end;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
ReguTerm1 = (lambda/m)*Theta1;
ReguTerm1(:,1)=0;
Theta1_grad = Theta1_grad + ReguTerm1;
ReguTerm2 = (lambda/m)*Theta2;
ReguTerm2(:,1)=0;
Theta2_grad = Theta2_grad + ReguTerm2;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
