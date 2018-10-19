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

C = 5;

%%%%%%
[svlist, b1] = solveSVM(X_t, y_t, C);
y_pred1 = prediction(X_v, svlist, b1);
acc1 = accuracy(y_v, y_pred1);
%%%%%%

%%%%%%
[svlist, b2] = solveSVM2(X_t, y_t, C);
y_pred2 = prediction(X_v, svlist, b2);
acc2 = accuracy(y_v, y_pred2);
%%%%%%


function [svlist, b] = solveSVM2(X, y, C)
% use quadratic programming to solve linear SVM
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   svlist: a list of support vectors
%   b: intercept, float scalar

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
            summation = summation + svlist(i).alpha * svlist(i).y * svlist(i).x' * svlist(k).x;
            %svlist(i).print()
        end
        bs = [bs, svlist(k).y - summation];
    end
    b = (min(bs) + max(bs))/2;
end


function [svlist, b] = solveSVM(X, y, C)
% use quadratic programming to solve linear SVM
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   svlist: a list of support vectors
%   b: intercept, float scalar

    [d, n] = size(X);    % dimension = d, number of rows = n
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