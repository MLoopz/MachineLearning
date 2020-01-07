function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% The collaborative filtering cost function (without regularization) 
%Hence, sum(sum(R.*M)) is the sum of all the elements of M for which the corresponding element in R equals 1.
%FuncionDeCosteJ= 1/2*sum(r(i,j)=1)(theta'*x - y)^2
%theta*X y no al revés para q se pueda multiplicar
J = (1/2)*sum(sum(((X*Theta' - Y) .^ 2) .* R)); %Multiplicamos por R para tener valor en posicones con 1 y 0 en 0;

% implement the gradient and use the checkCostFunction routine to check that the gradient is correct. 
%you only need to be concerned about the users who had given ratings to the movie, and this allows you to remove all the other users from Theta and Y.
%Concretely, you can set idx = find(R(i,:)==1) to be a list of all the users that have rated movie .
%ES LO MISMO USAR IDX que MULTIPLICAR POR R 
X_grad = ((X * Theta') - Y) .* R * Theta; %CAlulamos X para cada pelicula
Theta_grad = (((X * Theta') - Y) .* R)' * X;  % Calulamos theta para cada usuario 
% Finally, you should implement regularization.
regularization_termTheta= (lambda / 2) * sum(sum(Theta .^ 2));
regularization_termX= (lambda / 2) * sum(sum(X .^ 2));

J= J + regularization_termX + regularization_termTheta;
%This means that you just need to add lambda*x to the X_grad(i,:).
%Como es vectorizado no hace falta poner la i
X_grad = X_grad + (lambda.*X);
Theta_grad = Theta_grad + (lambda.*Theta);

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
