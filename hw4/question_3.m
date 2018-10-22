%run('/Users/caitao/Downloads/vlfeat-0.9.21/toolbox/vl_setup');

data = load('trainAnno.mat');
unAnno = data.ubAnno;

%HW4_Utils.demo1();
%HW4_Utils.demo2();

%{
[trD, trLb, valD, valLb, trRegs, valRegs, startNegDtrain, startNegDval] = HW4_Utils.getPosAndRandomNeg();
C = 10;
[w, b, obj, alpha] = svm_solve(trD, trLb, C);
HW4_Utils.genRsltFile(w, b, 'val', 'myoutfile');
[ap2, prec, rec] = HW4_Utils.cmpAP('myoutfile', 'val');
%}

C = 10;
[objs, aps, w, b] = hard_negative_mining(C);
csvwrite('objective2.csv', objs');
csvwrite('average_precision2.csv', aps');
HW4_Utils.genRsltFile(w, b, 'test', 'testoutfile');


function [objs, aps, w, b] = hard_negative_mining(C)
% Hard negative mining
% Args:
%   C: hyperparameter
% Return:
%   objs: a list of objective values
    objs = [];
    aps = [];
    [trD, trLb, valD, valLb, trRegs, valRegs, startNegDtrain, startNegDval] = HW4_Utils.getPosAndRandomNeg();
    [w, b, obj, alpha] = svm_solve(trD, trLb, C);
    HW4_Utils.genRsltFile(w, b, 'val', 'myoutfile');
    [ap, prec, rec] = HW4_Utils.cmpAP('myoutfile', 'val');
    objs = [objs, obj];
    aps = [aps, ap];
    %fprintf('%s', num2str(obj));
    for i = 1:10
        fprintf('i = %s\n', num2str(i));
        A = getA(trD, startNegDtrain, alpha);
        B = getB(w, b);
        progress = test_progress(objs);
        [trD, trLb] = update_trainDataLb(trD, trLb, A, B, progress);
        [w, b, obj, alpha] = svm_solve(trD, trLb, C);
        HW4_Utils.genRsltFile(w, b, 'val', 'myoutfile');
        [ap, prec, rec] = HW4_Utils.cmpAP('myoutfile', 'val');
        objs = [objs, obj];
        aps = [aps, ap];
    end
end


function progress = test_progress(objs)
% See whether there is progress
% Args:
%   objs: a list of objective values
% Return:
%   boolean value
    [~, n] = size(objs);
    progress = 1;
    if n < 2
        progress = 1;
    else
        if objs(n) > objs(n-1)
            progress = 1;
        end
    end
end


function A = getA(trD, startNegD, alpha)
% Find A: all non-support vectors in NegD. Non-support vectors: slack == 0
% Args:
%   trD: training dataset
%   startNegD: the index where NegD start
%   alpha: a list of slack variables
% Return:
%   a list of boolean

    A = [];
    [~, n] = size(trD);
    for i = 1:(startNegD-1)
        A = [A, 0];                   % positive patches, in ~A, won't be removed
    end
    for i = startNegD:n
        if alpha(i) < eps('single')
            A = [A, 1];               % non-support vector, in ~A, will be removed
        else
            A = [A, 0];               % support vector, in ~A, won't be removed
        end
    end
end


function hnrects = getB(w, b)
% Find hardest negative examples in the images
% Args:
%   w: weights, (d, 1)
%   b: intercept, float scalar
% Return:
%   hard negative samples. (6, n), a list, each element in list is [rect; imageIndex; slack]

    B = [];
    % step 1: detect
    imFiles = ml_getFilesInDir(sprintf('%s/%sIms/', HW4_Utils.dataDir, 'train'), 'jpg');
    nIm = length(imFiles);
    rects = cell(1, nIm);
    startT = tic;
    for i=1:nIm
        ml_progressBar(i, nIm, 'Ub detection', startT);
        im = imread(imFiles{i});
        rects{i} = HW4_Utils.detect(im, w, b);
    end

    % step 2: get the labels
    load(sprintf('%s/%sAnno.mat', HW4_Utils.dataDir, 'train'), 'ubAnno');    
    nIm = length(ubAnno);
    [detScores, isTruePos] = deal(cell(1, nIm));

    for i=1:nIm
        rects_i = rects{i};
        detScores{i} = rects_i(5,:);
        ubs_i = ubAnno{i}; % annotated upper body
        isTruePos_i = -ones(1, size(rects_i, 2));
        for j=1:size(ubs_i,2)
            ub = ubs_i(:,j);
            overlap = HW4_Utils.rectOverlap(rects_i, ub);
            isTruePos_i(overlap >= 0.5) = 1;
        end
        isTruePos{i} = isTruePos_i;
    end
    
    % step 3: compute slack of the negative samples
    hnrects = [];
    for i = 1:nIm
        rects_i = rects{i};
        isTruePos_i = isTruePos{i};
        [~, m] = size(rects_i);
        for j = 1:m
            if isTruePos_i(j) == -1         % negative patches
                slack = 1 + rects_i(5, j);  % 1 - (-1).*rects_i(j)
                if slack > 1                % violate the SVM margin, i.e. wrong classification
                    hn = [rects_i(1:4, j); i; slack];
                    hnrects = [hnrects, hn];
                end
            end
        end
    end
end


function [trD, trLb] = update_trainDataLb(trD, trLb, A, B, progress)
% Update the training data and labels
% Args:
%   trD: training features, (d, n)
%   trLb: training labels, (n, 1)
%   A: a list of index, non-support vectors in NegD
%   B: negative examples
% Return:
%   trD: updated training dataset
%   trLb: updated labels
    
    if progress == 1
        trD = trD(:, ~A);   % remove non-support vectors in NegD
        trLb = trLb(~A, :); % remove the corresbonding labels
    end

    im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, 'train', 1));
    [imH, imW, ~] = size(im);
    outofbound = [];
    [~, n] = size(B);
    for i=1:n           % remove rects that do not lie within image boundaries
        if B(3, i) > imW || B(4, i) > imH
            outofbound = [outofbound, 1];
        else
            outofbound = [outofbound, 0];
        end
    end
    B = B(:, ~outofbound);
    
    m = 548;            % the m in the slide
    [~, n] = size(trD);
    newhn = m - n;
    [~, n] = size(B);
    slacks = B(6, :);
    threshhold = max(slacks);
    if n > m            % add no more than 1000 negative training examples each iteration
        slacks = sort(slacks, 'descend');
        threshhold = slacks(newhn);
    end
    
    hardnegative = [];
    for i=1:n
        imIndex = B(5, i);
        im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, 'train', imIndex));
        [imH, imW, ~] = size(im);
        if B(6, i) < threshhold
            continue    % slack is below threshhold
        end
        imReg = im(round(B(2,i)):round(B(4,i)), round(B(1,i)):round(B(3,i)),:);
        imReg = imresize(imReg, HW4_Utils.normImSz);
        D_i = HW4_Utils.cmpFeat(rgb2gray(imReg));
        hardnegative = [hardnegative, D_i];
        trLb = [trLb; -1];
    end
    hardnegative = HW4_Utils.l2Norm(double(hardnegative));
    trD = [trD, hardnegative];
end



function [w, b, obj, alpha] = svm_solve(X, y, C)
% use quadratic programming to solve linear SVM
% Args:
%   X: features (d, n)
%   y: labels (n, 1)
%   C: 0 <= alpha <= C
% Return:
%   w: weights, (n, 1)
%   b: intercept, float scalar
%   obj: primal objective value

    [~, n] = size(X);    % dimension = d, number of rows = n
    f = -ones(n, 1);
    H = (X'*X).*(y*y');
    Aeq = y';            % equality constraint
    beq = 0;
    lb = zeros(n, 1);    % bound constraint
    ub(1:n, 1:1) = C;
    A = [];
    b = [];
    alpha = quadprog(H, f, A, b, Aeq, beq, lb, ub);

    %{
    svlist = [];
    for i = 1:n
        if alpha(i) > eps('single')
            svlist = [svlist, SupportVector(alpha(i), y(i), X(:, i))];
        end
    end
    %}

    [w, b] = compute_w_b(X, y, alpha);
    
    obj = objective_primal(X, y, w, b, C);
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
