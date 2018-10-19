% Solves question 2.6

%data = load('q2_2_data.mat');
%X_tr  = data.trD;    % training
%y_tr  = data.trLb;
%X_va  = data.valD;   % validation
%y_va  = data.valLb;
%X_te  = data.tstD;   % testing


data = load('q2_1_data.mat');
X_t  = data.trD;    % training
y_t  = data.trLb;
X_v  = data.valD;   % validation
y_v  = data.valLb;

C = 10;

%%%%%%
[svlist, b] = solveSVMlinear(X_t, y_t, C);
y_pred = prediction(X_v, svlist, b);
acc = accuracy(y_v, y_pred);
%%%%%%

%%%%%%
gamma = gamma_start(X_t);
[svlist, b2] = solveSVMrdf(X_t, y_t, C, gamma);
y_pred = prediction(X_v, svlist, b2);
acc2 = accuracy(y_v, y_pred);
%%%%%%

%{
gammas = linspace(0.1, 1, 20);
for i = 1:20
    [svlist, b2] = solveSVMrdf(X_t, y_t, C, gammas(i));
    y_pred = prediction(X_v, svlist, b2);
    acc2 = accuracy(y_v, y_pred);
    fprintf('gamma = %s, accuracy = %s\n', num2str(gammas(i)), num2str(acc2));
end
%}

function [svlist, b] = solveSVMlinear(X, y, C)
% use quadratic programming to solve SVM with linear kernel
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


function [svlist, b] = solveSVMrdf(X, y, C, gamma)
% use quadratic programming to solve SVM with linear kernel
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
    for i = 1:n
        for j = (i+1):n
            sub = X(:, i) - X(:, j);
            gammas = [gammas, sub' * sub];
        end
    end
    gamma = mean(gammas);
end


function pred = prediction(X, svlist, b)
% Prediction using alpha's and b, labels
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
            summation = summation + svlist(j).alpha * svlist(j).y * svlist(j).x' * X(:, i);
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