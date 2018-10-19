classdef SupportVector
    %Encapsulate a support vector
    properties
        alpha   % float
        y       % float
        x       % column vector (n, 1)
    end
    
    methods
        function obj = SupportVector(alpha, y, x)
            obj.alpha = alpha;
            obj.y = y;
            obj.x = x;
        end
    end

end

