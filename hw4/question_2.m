data = load('q2_1_data.mat');
X     = data.trD;    % training
y     = data.trLb;
X_val = data.valD;   % validation
y_val = data.valLb;

[d, n] = size(X);    % dimension = d, number of rows = n

f = -ones(n, 1);
H = (X'*X).*(y*y');
Aeq = y';            % equality constraint
beq = 0;
C = 0.1;
lb = zeros(n, 1);    % bound constraint
ub(1:n, 1:1) = C;
A = [];
b = [];

alpha = quadprog(H,f,A,b,Aeq,beq,lb,ub);
