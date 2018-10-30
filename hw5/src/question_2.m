X = load('../digit/digit.txt');
Y = load('../digit/labels.txt');


%{
% Question 2.5.2
k = 6;
[C, mu, i] = k_means(X, k);
sos = total_within_group_sum_of_squares(X, C, mu);
[p1, p2, p3] = pair_count_measure(Y, C);
%}


% Question 2.5.3 ~ 2.5.4
rng(0);       % this is a good random seed. It let k=4 have the best results
repeat = 10;
sos_list = [];
p1_list  = [];
p2_list  = [];
p3_list  = [];
for k = 1:10
    sos_sum = 0;
    p1_sum  = 0;
    p2_sum  = 0;
    p3_sum  = 0;
    for r = 1:repeat    % repeat several times for each k
        fprintf('k = %s, r = %s\n', num2str(k), num2str(r));
        
        [C, mu, i] = k_means(X, k);
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


csvwrite('../plot_data/p1.csv', p1_list');
csvwrite('../plot_data/p2.csv', p2_list');
csvwrite('../plot_data/p3.csv', p3_list');
csvwrite('../plot_data/sos.csv', sos_list');

