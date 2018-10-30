function [trainK, testK] = cmpExpX2Kernel2(trainD, testD, gamma)
% Precomputed kernel for SVM version 2
% Args:
%   trainD: training feature dataset, (n, d)
%   testD:  testing feature dataset, (n, d)
%   gamma:  hyper parameter, float
% Return:
%   trainK: precomputed kernel for training dateset
%   testK:  precomputed kernel for testing dataset

    [n, ~] = size(trainD);             % n is the # of samples in training set
    trainK = [];
    for i = 1:n
        x = trainD(i, :);
        kernel_i = [];
        for j = 1:n
            y = trainD(j, :);
            kernel_ij = exp_kernel(x, y, gamma);
            kernel_i = [kernel_i, kernel_ij];
        end
        trainK = [trainK; kernel_i];   % shape = (n, n)
    end
    trainK = [(1:n)', trainK];         % shape = (n, n+1)
    trainK = double(trainK);
    
    [m, ~] = size(testD);              % m is the # of samples in testing set
    testK = [];
    for i = 1:m
        x = testD(i, :);
        kernel_i = [];
        for j = 1:n
            y = trainD(j, :);
            kernel_ij = exp_kernel(x, y, gamma);
            kernel_i = [kernel_i, kernel_ij];
        end
        testK = [testK; kernel_i];     % shape = (m, n)
    end
    testK = [(1:m)', testK];           % shape = (m, n+1)
    testK = double(testK);
end


function kernel = exp_kernel(x, y, gamma)
% Compute the exponential \chi kernel between the ith and jth sample, i.e. K(Xi, Xj)
% Args:
%   X: feature dataset, (n, d)
%   i: the ith sample
%   j: the jth sample
% Return:
%   kernel: float

    d = length(x);
    kernel = 0;
    for k = 1:d
        kernel = kernel + (x(k)- y(k))^2 / (x(k) + y(k) + eps('single'));
    end
    kernel = exp(kernel * (-1/gamma));
end


