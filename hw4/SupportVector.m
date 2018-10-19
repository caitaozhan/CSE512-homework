classdef SupportVector
    %Encapsulate a support vector
    properties
        alpha   % float
        y       % float
        x       % column vector (n, 1)
    end
    
    methods
        function obj = SupportVector(alpha, y, x)
            % Init
            obj.alpha = alpha;
            obj.y = y;
            obj.x = x;
        end
        function print(obj)
            % print information
            fprintf('alpha = %s, y = %s\n', num2str(obj.alpha), num2str(obj.y));
        end
    end
end

