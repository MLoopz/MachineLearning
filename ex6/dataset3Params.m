function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%For C and sigma. we take values in multiplicative steps (e.g., 0.01, 0,03, 0.1, 0.3, 1, 3, 10, 30) for all possible pairs of values for C an sigma.
%predictions is a vector containing all the predictions from the SVM
%predicitons_errors is a vector with all the errors defined as the fraction of the cross validation examples that were classified incorrectly.
predictions_errors = zeros(8, 8); %8x8 = 64

arrayC=[0.01 0.03 0.1 0.3 1 3 10 30];
arraySigma=[0.01 0.03 0.1 0.3 1 3 10 30];

for i= 1:length(arrayC)
    for j= 1:length(arraySigma)
        %cogemos la posicion i del arrayC para C y posicion j de arraySigma para Sigma
        C=arrayC(i);
        sigma=arraySigma(j);
        
        model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
        predictions=svmPredict(model,Xval);
        predictions_errors(i,j) = mean(double(predictions ~= yval));
    end
end
fprintf('predictions_errors');
disp(predictions_errors);
[bestError_row,row_position]=min(predictions_errors);%buscamos el minimo de una matriz 8x8, nos defuelve un vector con los minimos de cada col y su row
[bestError_col,col_position]=min(bestError_row);%buscamos el minimo de una matriz 1x8, nos defuelve el minimo de la row y su col
%bestError=minimo de todas las rows y todas las cols
bestError=bestError_col;
bestC=arrayC(row_position(col_position)); %cogemos el row row_position con valores mínimo, referido a esa posicion dentro del arrayC
bestSigma=arraySigma(col_position); %cogemos la col col_position con valor mínimo, referido a esa posicion dentro del arraySigma

fprintf('bestError');
disp(bestError);
fprintf('bestS');
disp(bestSigma);
fprintf('bestC');
disp(bestC);

sigma=bestSigma;
C=bestC;
% =========================================================================

end
