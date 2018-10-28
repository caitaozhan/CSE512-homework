function [C_new, mu, i] = k_means(X, k)
% K means clustering
% Args:
%   k: number of clusters
%   X: feature data, (n, d)
% Return:
%   C_new: the label each sample is assigned to, (n, 1)
%   mu:    centers, (k, d)
%   i:     the number of iterations

    %mu = first_k_init_mu(X, k);  % Init center with the first k points in the dataset
    mu = rand_init_mu(X, k);      % Random Init
    max_iter = 20;
    C_old  = [];
    for i = 1:max_iter
        %fprintf('iter = %s\n', num2str(i));
        C_new  = classify(X, mu);  % coordinate descent on C
        mu = recenter(X, C_new);       % coordinate descent on mu
        
        bool = check_convergence(C_old, C_new);
        if bool == true
            return
        else
            C_old = C_new;
        end
        
    end
end

function bool = check_convergence(C_old, C_new)
% Check convergence: no change in label assignment from one step to another
% Args:
%   C_old: labels from the last iteration
%   C_new: labels from the new iteration
% Return:
%   bool: whether converge or not

    bool = true;
    if isempty(C_old) == true
        bool = false;
        return
    end
    [~, n] = size(C_old);
    for i = 1:n
        if C_old(i) ~= C_new(i)
            bool = false;
            break
        end
    end
end

function mu = first_k_init_mu(X, k)
% Init the centers with the first k points in the dataset
% Args:
%   X: feature data, (n, d)
%   k: number of centers
% Return:
%   mu: centers, (k, d)

    mu = [];
    for i = 1:k
        mu = [mu; X(i, :)];
    end
end


function mu = rand_init_mu(X, k)
% Choose k points from the dataset at random
% Args:
%   X: feature data, (n, d)
%   k: number of centers
% Return:
%   mu: centers, (k, d)

    [n, ~] = size(X);
    perm = randperm(n, k);
    mu = [];
    for i = 1:k
        mu = [mu; X(perm(i), :)];
    end
end


function C = classify(X, mu)
% Classify each point to the nearest center
% In the cooridinate descent, this is fix mu and minimize C
% Args:
%   X:  feature data, (n, d)
%   mu: centers, (2d array)
% Return:
%   C:  the label each sample is assigned to, (n, 1)

    C = [];
    [k, ~] = size(mu);
    [n, ~] = size(X);
    for i = 1:n
        mini = 999999999;
        label = 1;
        for j = 1:k
            dis = pdist([X(i, :); mu(j, :)], 'euclidean');
            if dis < mini
                mini = dis;
                label = j;
            end
        end
        C = [C, label];
    end
end


function mu = recenter(X, C)
% Recenter the mu
% In the cooridinate descent, this is fix C and minimzie mu
% Args:
%   X: feature data, (n, d)
%   C: the label each sample is assigned to, (n, 1)
% Return:
%   mu: centers, (k, d)

    k = length(unique(C));
    [n, d] = size(X);
    mu = [];
    for j = 1:k
        summation = zeros(1, d);
        counter = 0;
        for i = 1:n
            if C(i) == j
                summation = summation + X(i, :);
                counter = counter + 1;
            end
        end
        mu = [mu; summation/counter];
    end
end
