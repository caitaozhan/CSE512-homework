%run('/Users/caitao/Downloads/vlfeat-0.9.21/toolbox/vl_setup');

data = load('trainAnno.mat');
unAnno = data.ubAnno;

%HW4_Utils.demo1();
%HW4_Utils.demo2();

[trD, trLb, valD, valLb, trRegs, valRegs] = HW4_Utils.getPosAndRandomNeg();

%C = 10;
%[alpha, svlist] = solveSVM(trD, trLb, C);
%[w, b] = compute_w_b(trD, trLb, alpha);
%HW4_Utils.genRsltFile(w, b, 'val', 'myoutfile');
[ap, prec, rec] = HW4_Utils.cmpAP('myoutfile', 'val');


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
