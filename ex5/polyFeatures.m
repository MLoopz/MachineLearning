function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 

% the function should return a  matrix X_poly, where column 1 holds the original values of X, column 2 holds the values of X.^2, column 3 holds the values of X.^3, and so on.
for i = 1:p
    X_poly(:,i) = X(:,1).^i;  %primera columna de la matriz X a la potencia i(p en esa posicion)-> en la columna i de X_poly
end

% =========================================================================

end
