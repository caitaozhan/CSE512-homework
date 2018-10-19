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

    methods (Static)
        function ker = linear_kernel(xi, xj)
            % Linear kernel
            % Args:
            %   xi: column vector, (n, 1)
            %   xj: column vector, (n, 1)
            % Return:
            %   float scalar
            ker = xi' * xj;
        end
        function ker = rdf_kernel(xi, xj, gamma)
            % Radial basis function kernel, i.e. Gaussian kernel
            % Args:
            %   xi: column vector, (n, 1)
            %   xj: column vector, (n, 1)
            % Return:
            %   float scalar
            sub = xi - xj;
            ker = exp(-(sub' * sub)/gamma);
        end
    end     
end

