% Solves question 2.1 ~ 2.5

data = load('q2_1_data.mat');
X_t  = data.trD;    % training
y_t  = data.trLb;
X_v  = data.valD;   % validation
y_v  = data.valLb;

% Training %
C = 0.1;
[alpha, svlist] = solveSVM(X_t, y_t, C);
[w, b] = compute_w_b(X_t, y_t, alpha);
%%%%%%%%%%%%

% Accuracy %
y_pred1 = prediction(X_v, svlist, b);
y_pred2 = prediction2(X_v, w, b);
acc1 = accuracy(y_v, y_pred1);
acc2 = accuracy(y_v, y_pred2);
fprintf('The accuracy on the validation dataset = %s \n', num2str(acc1))
%%%%%%%%%%%%

% Objective %
obj_t_p = objective_primal(X_v, y_v, w, b, C);
fprintf('The objective value = %s \n', num2str(obj_t_p));

%obj_t_p = objective_primal(X_t, y_t, w, b, C);
%obj_t_d = objective_dual(X_t, y_t, alpha);
%fprintf('Objective train primal = %s\n', num2str(obj_t_p));
%fprintf('Objective train dual   = %s\n', num2str(obj_t_d));
%%%%%%%%%%%%%

% Number of support vectors %
[~, num] = size(svlist);
fprintf('Number of support vectors = %s\n', num2str(num));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Confusion Matrix %
y_true = transform_label(y_v');
y_pred = transform_label(y_pred1);
plotconfusion(y_true, y_pred);
%%%%%%%%%%%%%%%%%%%%


function y_trans = transform_label(y)
% Transform the lables from {-1, 1} into {0, 1}
% Args:
%   y: (1, n)
% Return:
%   y_trans: (1, n)
    y_trans = [];
    [~, n] = size(y);
    for i = 1:n
        if y(i) == -1
            y_trans = [y_trans, 0];
        else
            y_trans = [y_trans, 1];
        end
    end
end


function obj = objective_primal(X, y, w, b, C)
% Compute the primal objective of a (linear) SVM
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   w: weights (n, 1)
%   b: intercept, float scalar
% Return:
%   obj: float scalar
    [~, n] = size(X);
    obj = 0.5 * (w') * w;
    for i = 1:n
        slack = 1 - y(i) * (w' * X(:, i) + b);
        if slack < 0
            slack = 0;
        end
        obj = obj + C * slack;
    end
end


function obj = objective_dual(X, y, alpha)
% Compute the dual objective of a (linear) SVM
% Note: this can only be used on the same data as trained
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   alpha: Lagrangian multipliers (n, 1)
% Return:
%   obj: float scalar
    [~, n] = size(X);
    f = ones(n, 1);
    H = (X'*X).*(y*y');
    obj = f'*alpha - 0.5 * alpha' * H * alpha;
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


function pred = prediction2(X, w, b)
% Prediction using w and b, labels
% Time complexity: O(nd)
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   w: weights (n, 1)
%   b: intercept, float scalar
% Return:
%   pred: (n, 1)
    [~, n] = size(X);
    pred = [];
    for i = 1:n
        y = w' * X(:, i) + b;
        if y >= 0
            pred = [pred, 1];
        else
            pred = [pred, -1];
        end
    end
end


function pred = prediction_score(X, svlist, b)
% Prediction using alpha's and b, real score value
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
        pred = [pred, summation];
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


function [alpha, svlist] = solveSVM(X, y, C)
% use quadratic programming to solve linear SVM
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   alpha: Lagrangian multipliers (n, 1)
%   svlist: a list of support vectors

    [~, n] = size(X);    % dimension = d, number of rows = n
    f = -ones(n, 1);
    H = (X'*X).*(y*y');
    Aeq = y';            % equality constraint
    beq = 0;
    lb = zeros(n, 1);    % bound constraint
    ub(1:n, 1:1) = C;
    A = [];
    b = [];
    alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);
    
    svlist = [];
    for i = 1:n
        if alpha(i) > eps('single')
            svlist = [svlist, SupportVector(alpha(i), y(i), X(:, i))];
        end
    end
end


function [w, b] = compute_w_b(X, y, alpha)
% Compute w and b of the primal from alpha of the dual
% Time complexity: O(nd)
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   alpha: Lagrangian multipliers
% Return:
%   w: weights (d, 1)
%   b: intercept, float scalar

    [d, n] = size(X);
    w = zeros(d, 1);
    for i = 1:n
        w = w + alpha(i) * y(i) * X(:, i);
    end
    mymin = 999999999;
    mymax = -mymin;
    for i = 1:n
        temp = w' * X(:, i);
        %fprintf('%s \n', num2str(temp))
        if y(i) == 1
            mymin = min(mymin, temp);
        elseif y(i) == -1
            mymax = max(mymax, temp);
        else
            fprintf('Error in the labels');
        end
    end
    b = -(mymax + mymin)/2;
end

