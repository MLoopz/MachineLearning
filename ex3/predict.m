function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
%---------------------------------------------------------------------------------
% Cada posicion/pixel de la imagen es una feature. Tenemos 400 features por lo que tenemos 400 nodos. 
% Tenemos 5000 imagens enel dataset
%---------------------------------------------------------------------------------
% layer1 = 400 nodos + 1bias
% layer2 = 25 nodos + 1bias 
% layer3 = 10 nodos
%---------------------------------------------------------------------------------
% theta1 = 25 nodos en la hidden layer x 401 posicions en cada imagen (20*20) + 1 bias
% theta2 = 10 nodos en la tercera(output) layer x 26 (25 nodos) + 1 bias
%---------------------------------------------------------------------------------

a1 = [ones(m,1) X]; % 5000 x 401. Añadiendo 1 columna en X de bias
z2 = a1 * Theta1';  % 5000 x 25. Sin activar
a2 = sigmoid(z2);   % 5000 x 25. Activacion de z con g es a

a2 =  [ones(size(a2,1),1) a2];  % 5000 x 26. Añadimos columna de bias en a2
z3 = a2 * Theta2';  % 5000 x 10. Sin activar
a3 = sigmoid(z3);  % 5000 x 10. Activacion de z con g es a

[prob, p] = max(a3,[],2); % m  x 1  to obtain the max for each row. "2" para devolver el max en la dimension de la fila(1) 1 x fila y no de la col(1)(1 xcol)

% =========================================================================


end
