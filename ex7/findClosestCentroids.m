function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

%Assigning each training example to its closest centroid

% Initialize centroids
%centroids = kMeansInitCentroids(X, K); Se hace fuera

n_points=size(X,1); %tamaño de una columna de X (nº puntos/ejemplos)
centroid_prox=zeros(n_points,1); %para cada centroide, distancia y posicion en el array de ese centroide
for i = 1:n_points
    distancias=zeros(K,1);
    for j = 1:K %numero de centroides de clusters
        distancias(j) = sqrt(sum( (X(i,:) - centroids(j,:)) .^2 ));
    end
    %nos quedamos con la posicion donde estala minima distancia para esa punto(i) a todos los centroides
    [distancia,centroid_prox(i)]=min(distancias);%[distancia,posicion en el array(centroide)]
end
idx=centroid_prox;
% =============================================================

end

