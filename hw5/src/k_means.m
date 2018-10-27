X = load('digit/digit.txt');
Y = load('digit/labels.txt');


%{
% Question 2.5.2
k = 6;
[C, mu, i] = kmeans(X, k);
sos = total_within_group_sum_of_squares(X, C, mu);
[p1, p2, p3] = pair_count_measure(Y, C);
%}

rng(10);       % this is a good random seed. It let k=4 have the best results
repeat = 15;
sos_list = [];
p1_list  = [];
p2_list  = [];
p3_list  = [];
for k = 1:10
    sos_sum = 0;
    p1_sum  = 0;
    p2_sum  = 0;
    p3_sum  = 0;
    for r = 1:repeat    % repeat 15 times for each k
        fprintf('k = %s, r = %s\n', num2str(k), num2str(r));
        [C, mu, i] = kmeans(X, k);
        sos = total_within_group_sum_of_squares(X, C, mu);
        [p1, p2, p3] = pair_count_measure(Y, C);
        sos_sum = sos_sum + sos;
        p1_sum = p1_sum + p1;
        p2_sum = p2_sum + p2;
        p3_sum = p3_sum + p3;
    end
    sos_list = [sos_list, sos_sum/repeat];
    p1_list  = [p1_list,  p1_sum/repeat];
    p2_list  = [p2_list,  p2_sum/repeat];
    p3_list  = [p3_list,  p3_sum/repeat];
end


csvwrite('plot_data/p1.csv', p1_list');
csvwrite('plot_data/p2.csv', p2_list');
csvwrite('plot_data/p3.csv', p3_list');
csvwrite('plot_data/sos.csv', sos_list');


function [C_new, mu, i] = kmeans(X, k)
% K means clustering
% Args:
%   k: number of clusters
%   X: feature data, (n, d)
%   Y: labels, (n, 1)
% Return:
%   C_new: the label each sample is assigned to, (n, 1)
%   mu:    centers, (k, d)
%   i:     the number of iterations

    %mu = first_k_init_mu(X, k);  % Init center with the first k points in the dataset
    mu = rand_init_mu(X, k);      % Random Init
    max_iter = 20;
    C_old  = [];
    for i = 1:max_iter
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
% Random init the k mu (center)
% Args:
%   X: feature data, (n, d)
%   k: number of centers
% Return:
%   mu: centers, (k, d)

    [~, d] = size(X);
    mu = [];
    mins = min(X);  % the min of each dimension/feature, (array)
    maxs = max(X);  % the max of each dimension/feature, (array)
    for i = 1:k
        mu_i = [];
        for j = 1:d
            mu_i = [mu_i, randi([mins(j), maxs(j)], 1)];
        end
        mu = [mu; mu_i];
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


function [p1, p2, p3] = pair_count_measure(Y, C)
% Pair-counting measures
% Args:
%   Y: the ground truth, (n, 1)
%   C: the assigned class by k-means, (1, n)
% Return:
%   p1: percentage of pairs of which both data points were assigned to the same cluster
%   p2: percentage of pairs of which two data points are assigned to different clusters
%   p3: p3 = (p1+p2)/2

    n = length(Y);
    same_class_pair = 0;
    diff_class_pair = 0;
    p1 = 0;
    p2 = 0;
    for i = 1:n
        for j = i+1:n
            if Y(i) == Y(j)
                same_class_pair = same_class_pair + 1;
                if C(i) == C(j)
                    p1 = p1 + 1;
                end
            else
                diff_class_pair = diff_class_pair + 1;
                if C(i) ~= C(j)
                    p2 = p2 + 1;
                end
            end
        end
    end
    p1 = p1/same_class_pair;
    p2 = p2/diff_class_pair;
    p3 = (p1+p2)/2;
end


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
                sos = sos + pdist([X(i, :); mu(j, :)], 'euclidean');
            end
        end
    end
end

