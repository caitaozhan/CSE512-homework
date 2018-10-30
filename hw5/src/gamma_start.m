function gamma = gamma_start(X)
% Get the starting point of gamma for rdf kernel
% Args:
%   X: feature data, (n, d)
% Return:
%   float scalar
    gammas = [];
    [~, n] = size(X);
    for i = 1:n
        for j = (i+1):n
            sub = X(i, :) - X(j, :);
            gammas = [gammas, sub * sub'];
        end
    end
    gamma = mean(gammas);
end