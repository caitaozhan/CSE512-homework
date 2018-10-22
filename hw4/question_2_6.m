% Solves question 2.6

%{
data = load('q2_1_data.mat');
X_t  = data.trD;    % training
y_t  = data.trLb;
X_v  = data.valD;   % validation
y_v  = data.valLb;

C = 10;

%%%%%%
[svlist, b] = solve_svm_linear(X_t, y_t, C);
y_pred = prediction_linear(X_v, svlist, b);
acc = accuracy(y_v, y_pred);
%%%%%%

%%%%%%
gamma = gamma_start(X_t);
[svlist, b2] = solve_svm_rdf(X_t, y_t, C, gamma);
y_pred = prediction_rdf(X_v, svlist, b2, gamma);
acc2 = accuracy(y_v, y_pred);
%%%%%%


gammas = linspace(0.1, 2, 20);
for i = 1:20
    [svlist, b2] = solve_svm_rdf(X_t, y_t, C, gammas(i));
    y_pred = prediction_rdf(X_v, svlist, b2, gammas(i));
    acc2 = accuracy(y_v, y_pred);
    fprintf('gamma = %s, accuracy = %s\n', num2str(gammas(i)), num2str(acc2));
end
%}

data = load('q2_2_data.mat');
X_t  = data.trD;       % training
y_t  = data.trLb;
X_v  = data.valD;      % validation
y_v  = data.valLb;
X_test  = data.tstD;   % testing


C = 1;
%binarySVMs = solve_svm_multi_class_linear(X_t, y_t, C);
y_pred = prediction_linear_multi(X_v, binarySVMs);
acc = accuracy(y_v, y_pred);
y_pred = prediction_linear_multi(X_test, binarySVMs);
csvwrite('pred_linear_C_1.csv', y_pred');

%{
C = 1;
gamma = gamma_start(X_t);
gamma = 5000;
binarySVMs = solve_svm_multi_class_rdf(X_t, y_t, C, gamma);
y_pred = prediction_rdf_multi(X_v, binarySVMs, gamma);
acc2 = accuracy(y_v, y_pred);
y_pred = prediction_rdf_multi(X_test, binarySVMs, gamma);
csvwrite('pred_rdf_C_100.csv', y_pred');
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function binarySVMs = solve_svm_multi_class_linear(X, y, C)
% use quadratic programming to solve multi-class SVM using one-versus-rest with linear kernel
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   a list of BinarySVM
    binarySVMs = [];
    [n, ~] = size(y);
    for label = 1:10               % there are 10 different labels from 1 to 10
        y_binary = ones(n, 1);
        for i = 1:n
            if y(i) ~= label
                y_binary(i) = -1;  % one-versus-rest
            end 
        end
        [svlist, b] = solve_svm_linear(X, y_binary, C);
        binarySVMs = [binarySVMs, BinarySVM(label, svlist, b)];
    end
end


function pred = prediction_linear_multi(X, binarySVMs)
% Prediction with linear kernal
% Args:
%   X: features (d, n)
%   binarySVMs: a list of BinarySVM
% Return:
%   pred: (n, 1)
    [~, n] = size(X);           % n number of sample
    [~, m] = size(binarySVMs);  % m number of BinarySVM
    pred = [];
    for i = 1:n        % A particular point is assigned to the class for which 
        maxi = -1;     % the distance from the margin, in the positive direction, is maximal
        pred_temp = 0;
        for j = 1:m
            label = binarySVMs(j).label;
            svlist = binarySVMs(j).svlist;
            b = binarySVMs(j).b;
            score = prediction_linear_score(X(:, i), svlist, b);
            if score > maxi
                pred_temp = label;
                maxi = score;
            end
        end
        pred = [pred, pred_temp];
    end
end


function score = prediction_linear_score(x, svlist, b)
% Prediction using alpha's and b, labels with linear kernel
% Time complexity: worst case O(n^2 * d), but usally O(nd*SV) where SV is
% the number of support vectors is usually << n
% Args:
%   x: features (d, 1), a single sample
%   y: labels (n, 1)
%   svlist: a list of support vectors
%   b: intercept, float
% Return:
%   a float scalar
    summation = 0;
    [~, n_sv] = size(svlist);
    for j = 1:n_sv
        summation = summation + svlist(j).alpha * svlist(j).y * SupportVector.linear_kernel(svlist(j).x, x);
    end
    score = summation + b;
end
%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%
function binarySVMs = solve_svm_multi_class_rdf(X, y, C, gamma)
% use quadratic programming to solve multi-class SVM using one-versus-rest with rdf kernel
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   a list of BinarySVM
    binarySVMs = [];
    [n, ~] = size(y);
    for label = 1:10               % there are 10 different labels from 1 to 10
        y_binary = ones(n, 1);
        for i = 1:n
            if y(i) ~= label
                y_binary(i) = -1;  % one-versus-rest
            end 
        end
        [svlist, b] = solve_svm_rdf(X, y_binary, C, gamma);
        binarySVMs = [binarySVMs, BinarySVM(label, svlist, b)];
    end
end


function pred = prediction_rdf_multi(X, binarySVMs, gamma)
% Prediction with rdf kernal
% Args:
%   X: features (d, n)
%   binarySVMs: a list of BinarySVM
% Return:
%   pred: (n, 1)
    [~, n] = size(X);           % n number of sample
    [~, m] = size(binarySVMs);  % m number of BinarySVM
    pred = [];
    for i = 1:n        % A particular point is assigned to the class for which 
        maxi = -1;     % the distance from the margin, in the positive direction, is maximal
        pred_temp = 0;
        for j = 1:m
            label = binarySVMs(j).label;
            svlist = binarySVMs(j).svlist;
            b = binarySVMs(j).b;
            score = prediction_rdf_score(X(:, i), svlist, b, gamma);
            if score > maxi
                pred_temp = label;
                maxi = score;
            end
        end
        pred = [pred, pred_temp];
    end
end


function score = prediction_rdf_score(x, svlist, b, gamma)
% Prediction using alpha's and b, labels with rdf kernel
% Time complexity: worst case O(n^2 * d), but usally O(nd*SV) where SV is
% the number of support vectors is usually << n
% Args:
%   x: features (d, 1), a single sample
%   y: labels (n, 1)
%   svlist: a list of support vectors
%   b: intercept, float
% Return:
%   a float scalar
    summation = 0;
    [~, n_sv] = size(svlist);
    for j = 1:n_sv
        summation = summation + svlist(j).alpha * svlist(j).y * SupportVector.rdf_kernel(svlist(j).x, x, gamma);
    end
    score = summation + b;
end
%%%%%%%%%%%%%%%%%%%%%%%



function [svlist, b] = solve_svm_linear(X, y, C)
% use quadratic programming to solve binary SVM with linear kernel
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   svlist: a list of support vectors
%   b: intercept, float scalar

    [~, n] = size(X);    % dimension = d, number of rows = n
    f = -ones(n, 1);
    H = zeros(n, n);
    for i = 1:n
        for j = 1:n
            H(i, j) = SupportVector.linear_kernel(X(:, i), X(:, j));
        end
    end
    H = H .* (y*y');
    Aeq = y';            % equality constraint
    beq = 0;
    lb = zeros(n, 1);    % bound constraint
    ub(1:n, 1:1) = C;
    A = [];
    b = [];
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    
    svlist = [];
    for i = 1:n
        if alpha(i) > eps('single')   % support vectors are those whose alpha > 0
            svlist = [svlist, SupportVector(alpha(i), y(i), X(:, i))];
        end
    end

    [~, n_sv] = size(svlist);
    fprintf('# of SV = %s\n', num2str(n_sv));

    bs = [];
    for k = 1:n_sv
        summation = 0;
        for i = 1:n_sv
            summation = summation + svlist(i).alpha * svlist(i).y * SupportVector.linear_kernel(svlist(i).x, svlist(k).x);
        end
        bs = [bs, svlist(k).y - summation];
    end
    b = (min(bs) + max(bs))/2;
end


function [svlist, b] = solve_svm_rdf(X, y, C, gamma)
% use quadratic programming to solve binary SVM with linear kernel
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   svlist: a list of support vectors
%   b: intercept, float scalar

    [~, n] = size(X);    % dimension = d, number of rows = n
    f = -ones(n, 1);
    H = zeros(n, n);
    for i = 1:n
        for j = 1:n
            H(i, j) = SupportVector.rdf_kernel(X(:, i), X(:, j), gamma);
        end
    end
    H = H .* (y*y');
    Aeq = y';            % equality constraint
    beq = 0;
    lb = zeros(n, 1);    % bound constraint
    ub(1:n, 1:1) = C;
    A = [];
    b = [];
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    
    svlist = [];
    for i = 1:n
        if alpha(i) > eps('single')   % support vectors are those whose alpha > 0
            svlist = [svlist, SupportVector(alpha(i), y(i), X(:, i))];
        end
    end

    [~, n_sv] = size(svlist);
    fprintf('# of SV = %s\n', num2str(n_sv));

    bs = [];
    for k = 1:n_sv
        summation = 0;
        for i = 1:n_sv
            summation = summation + svlist(i).alpha * svlist(i).y * SupportVector.rdf_kernel(svlist(i).x, svlist(k).x, gamma);
        end
        bs = [bs, svlist(k).y - summation];
    end
    b = (min(bs) + max(bs))/2;
end


function gamma = gamma_start(X)
% Get the starting point of gamma
% Args:
%   svlist: a list of SupportVector
% Return:
%   float scalar
    gammas = [];
    [~, n] = size(X);
    for i = 1:4:n
        for j = (i+1):7:n
            sub = X(:, i) - X(:, j);
            gammas = [gammas, sub' * sub];
        end
    end
    gamma = mean(gammas);
end


function pred = prediction_linear(X, svlist, b)
% Prediction using alpha's and b, labels with linear kernel
% Time complexity: worst case O(n^2 * d), but usally O(nd*SV) where SV is
% the number of support vectors is usually << n
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   svlist: a list of support vectors
%   b: intercept, float
% Return:
%   pred: (n, 1)

    [~, n] = size(X);
    pred = [];
    for i = 1:n
        summation = 0;
        [~, n_sv] = size(svlist);
        for j = 1:n_sv
            summation = summation + svlist(j).alpha * svlist(j).y * SupportVector.linear_kernel(svlist(j).x, X(:, i));
        end
        summation = summation + b;
        if summation >= 0
            pred = [pred, 1];
        else
            pred = [pred, -1];
        end
    end
end


function pred = prediction_rdf(X, svlist, b, gamma)
% Prediction using alpha's and b, labels with linear kernel
% Time complexity: worst case O(n^2 * d), but usally O(nd*SV) where SV is
% the number of support vectors is usually << n
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   svlist: a list of support vectors
%   b: intercept, float
% Return:
%   pred: (n, 1)

    [~, n] = size(X);
    pred = [];
    for i = 1:n
        summation = 0;
        [~, n_sv] = size(svlist);
        for j = 1:n_sv
            summation = summation + svlist(j).alpha * svlist(j).y * SupportVector.rdf_kernel(svlist(j).x, X(:, i), gamma);
        end
        summation = summation + b;
        if summation >= 0
            pred = [pred, 1];
        else
            pred = [pred, -1];
        end
    end
end


function acc = accuracy(y, y_pred)
% Compute the accuracy
% Args:
%   y: ground truth labels, (n, 1)
%   y_pred: prediction (n, 1)
% Return:
%   acc: float scalar
    [~, n] = size(y_pred);
    counter = 0;
    for i = 1:n
        if y(i) == y_pred(i)
            counter = counter + 1;
        end
    end
    acc = counter/n;
end
