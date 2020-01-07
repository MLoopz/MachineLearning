function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%

% epsilon=0.12. [-epsilon, epsilon] This range of values ensures that the parameters are kept small and makes the learning more efficient.
epsilon_init = 0.12;
% One effective strategy for choosing epsilon, is to base it on the number of units in the network. 
% A good choice of epsilon is:  
epsilon_init = sqrt(6)/(sqrt(L_in)+sqrt(L_out)); % where L_in=S (units in the layer) and L_out=S+1
disp(epsilon_init);
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;








% =========================================================================

end
