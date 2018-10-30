function gamma = gamma_start2(X)
% Get the default value of gamma for exponential kernel
% Args:
%   X: feature dataset
% Return:
%   gamma, float

    [n, ~] = size(X);
    gammas = [];
    for i = 1:n
        for j = (i+1):n
            chi_dis = chi_dist(X, i, j);
            gammas = [gammas, chi_dis];
        end
    end
    gamma = mean(gammas);
end


function chi_dis = chi_dist(X, i, j)
% Compute the chi distance between Xi and Xj
% Args:
%   X: feature data
%   i: ith sample
%   j: jth sample
% Return:
%   chi_dis: float
    x = X(i, :);
    y = X(j, :);
    d = length(x);
    chi_dis = 0;
    for k = 1:d
        chi_dis = chi_dis + (x(k)- y(k))^2 / (x(k) + y(k) + eps('single'));
    end
end