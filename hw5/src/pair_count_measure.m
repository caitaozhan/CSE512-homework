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
