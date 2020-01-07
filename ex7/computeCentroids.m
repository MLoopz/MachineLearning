function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%Recomputing the mean of each centroid using the points assigned to it. 
    indicesPorCentroide=zeros(K, m);
    for i = 1:K %numero de centroides de clusters
        %buscamos en la lista de centroides idx, todas las posiciones que
        %pertenecen a cada centroide
        for j = 1:m
            if(idx(j)==i)
                %guardamos la posicion del punto en idx que pertenece a ese
                %centroide i
                indicesPorCentroide(i,j)=j;
            end     
        end
        %the row vector centroids(i, :) should contain the mean of the data points assigned to centroid i.
        %array de ese row i
        array_indices=indicesPorCentroide(i,:);
        %quitamos zeros
        array_indices(array_indices==0)=[]; 
        %le pasamos el array de las posiciones sin zeros
        centroids(i,:)=mean(X(array_indices,:));%mean(X(indicesPorCentroide(i,:))); sin zeros
    end





% =============================================================


end

