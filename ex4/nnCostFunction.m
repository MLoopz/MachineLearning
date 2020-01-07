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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

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

%--------------- Part 1: Calculamos J -------------
% Theta1 = 25 x 401
% Theta2 = 10 x 26

X = [ones(m,1), X];  % Añadimos 1 en la columna como bias
%Forward propagation
a1 = X; % 5000 x 401 ->5000 ejemplos de 20x20 pixeles + bias
z2= a1 * Theta1'; %5000x25
a2= sigmoid(z2); %5000x25
%añadimos bias a a2
a2 = [ones(size(a2,1),1), a2]; %5000 x 26
z3= a2 * Theta2'; %5000x10
a3= sigmoid(z3); %5000*10
h=a3;

yK = (1:num_labels)==y; % vector de Y dónde cada row tiene un 1 en su posicion y(i)
%yK = zeros(m,num_labels)
% for i = 1:m
%     yK(i,y(i)) = 1;
% end
%[100000000]
%[100000000]
%[100000000]
%[010000000]
%[010000000]
%[010000000]
%...
% Costfunction sin regularizacion
J = (1/m) * sum(sum((-yK.*log(h))-((1-yK).*log(1-h))));  %coste J segun Theta

% Costfunction con regularizacion
% No regularizamos bias (first column of each matrix)
regularizacion = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))); % theta(todo, de la 2 a end)
J = J + regularizacion;

%--------------- Part 2: BackProp - Theta1_grad and Theta2_grad,  partial derivatives of the cost function -------------
%sigmoidGradient Hecho
%randomInitialization Hecho
%BackPropagation
   %1.Given a training example , we will first run a 'forward pass' to compute all the activations throughout the network,
   % including the output value of the hypothesis.
   % Then, for each node in layer , we would like to compute an 'error term'  that measures how much that node was 'responsible' for any errors in our output.
   % For an output node, we can directly measure the difference between the network's activation and the true target value, and use that to define  (since layer 3 is the output layer). For the hidden units, you will compute  based on a weighted average of the error terms of the nodes in layer .

for t=1:m    
   
  %1. calculamos activaciones a
    % capa 1
  a1 = X(t,:)'; % (n+1) x 1 == 401 x 1. Ya añadimos antes 1 en la priemra columan como bias
    % capa 2
  z2 = Theta1*a1;  % hidden_layer_size x 1 == 25 x 1
  a2 = sigmoid(z2); 
  a2 = [1; a2]; % (hidden_layer_size+1) x 1 == 26 x 1. añadimos columna Bias
    % capa 3
  z3 = Theta2 * a2; % num_labels x 1 == 10 x 1    
  a3 = sigmoid(z3); % num_labels x 1 == 10 x 1   
  
  %2. vector de salidas de y (Yk).Vector de Y donde cada clase "i" tiene un 1 en la posicion "i" de la matriz
  vector_y = (1:num_labels)'==y(t); % num_labels x 1 == 10 x 1.
    
  %3. calculamos deltas para cada layer
  %For an output node, we can directly measure the difference between the network's activation and the true target value= delta3
  %calculating delta values
  delta3 = a3 - vector_y; % num_labels x 1 == 10 x 1    
  delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)]; % (hidden_layer_size+1) x 1 == 26 x 1
  delta2 = delta2(2:end); % hidden_layer_size x 1 == 25 x 1 %Removing delta2 for bias node .Quitamos columna 1 de bias de delta2 (delta0 de layer 2)
  % delta_1 no se calcula ya que no hay error en el input
  
  %4. Accumulate the gradient from this example using the formula in
  %ThetaN_grad (variables para acumular)
  Theta1_grad = Theta1_grad + (delta2 * a1'); % 25 x 401
  Theta2_grad = Theta2_grad + (delta3 * a2'); % 10 x 26
end

%5. Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by m (formula)
Theta1_grad = (1/m) * Theta1_grad; % 25 x 401
Theta2_grad = (1/m) * Theta2_grad; % 10 x 26

%--------------- Part 3: Regularizatio in ThetaN_grad -------------

%Calculating gradients for the regularization
Theta1_grad_regularizacion = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
Theta2_grad_regularizacion = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
%Adding regularization term to earlier calculated Theta_grad
Theta1_grad = Theta1_grad + Theta1_grad_regularizacion;
Theta2_grad = Theta2_grad + Theta2_grad_regularizacion;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
