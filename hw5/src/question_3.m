%cv_accuracy = HW5_BoW.main();

rng(0);
scales = [8, 16, 32, 64];
normH = 16;
normW = 16;
%bowCs = HW5_BoW.learnDictionary(scales, normH, normW);

[trIds, trLbs] = ml_load('../bigbangtheory_v2/train.mat',  'imIds', 'lbs');             
tstIds = ml_load('../bigbangtheory_v2/test.mat', 'imIds'); 

%trD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
%trD = trD';
%tstD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
%tstD = tstD';

%3.4.2
%cv_accuracy = svmtrain(trLbs, trD, '-v 5');

%{
%3.4.3
C = [0.1, 1, 10, 20, 40, 80, 160];
G = [0.1, 1, 10, 20, 40, 80, 160];

for i = 1:length(C)
    for j = 1:length(G)
        options = sprintf('-c %d -g %d -v 5 -q', C(i), G(j));
        cv_accuracy = svmtrain(trLbs, trD, options);
        fprintf('options = %s, accuracy = %s\n', options, cv_accuracy);
    end
end
%}


G = [0.1, 0.4780, 1, 2];
C = [0.1, 1, 10, 20, 40, 80, 160];
for j = 1:length(G)
    [trainK] = cmpExpX2Kernel(trD, G(j));
    for i = 1:length(C)
        options = sprintf('-c %d -g %d -t 4 -v 5 -q', C(i), G(j));
        cv_accuracy = svmtrain(trLbs, trainK, options);
        fprintf('options = %s, accuracy = %s\n', options, cv_accuracy);
    end
end


gamma = gamma_start2(trD); % gamma = 0.4780



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Write code for training svm and prediction here            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [trainK] = cmpExpX2Kernel(trainD, gamma)
% Precomputed kernel for SVM
% Args:
%   trainD: training feature dataset, (n, d)
%   gamma:  hyper parameter, float
% Return:
%   trainK: precomputed kernel for training dateset

    [n, ~] = size(trainD);
    trainK = [];
    for i = 1:n
        kernel_i = [];
        for j = 1:n
            kernel_ij = exp_kernel(trainD, i, j, gamma);
            kernel_i = [kernel_i, kernel_ij];
        end
        trainK = [trainK; kernel_i];
    end
    trainK = [(1:n)', trainK];
    trainK = double(trainK);
end


function kernel = exp_kernel(X, i, j, gamma)
% Compute the exponential \chi kernel between the ith and jth sample, i.e. K(Xi, Xj)
% Args:
%   X: feature dataset, (n, d)
%   i: the ith sample
%   j: the jth sample
% Return:
%   kernel: float

    x = X(i, :);
    y = X(j, :);
    d = length(x);
    kernel = 0;
    for k = 1:d
        kernel = kernel + (x(k)- y(k))^2 / (x(k) + y(k) + eps('single'));
    end
    kernel = exp(kernel * (-1/gamma));
end

