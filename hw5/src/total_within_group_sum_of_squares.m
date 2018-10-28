function sos = total_within_group_sum_of_squares(X, C, mu)
% Compute total within group sum of squares
% Args:
%   X: feature data, (n, d)
%   C: the label each sample is assigned to, (n, 1)
%   mu: centers, (k, d)
% Return:
%   sos: total within group sum of squares

    [k, ~] = size(mu);
    [n, ~] = size(X);
    sos = 0;
    for j = 1:k
        for i = 1:n
            if C(i) == j
                sub = X(i, :) - mu(j, :);
                sos = sos + sub * sub';
            end
        end
    end
end





