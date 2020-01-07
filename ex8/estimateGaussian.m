function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = ((1 / m) * sum(X))'; % 1x10, X= 1682*10, mu=1x2 ' -> 2x1 La salida tiene que ser 2x1 como cuando se crea por lo que '
sigma2 = ((1 / m) * sum((X - mu') .^2))'; % X= 307x2,mu'=1x2 sum=307x1, m=1*1. La salida tiene que ser sigma=2x1 por lo que '

% =============================================================


end
