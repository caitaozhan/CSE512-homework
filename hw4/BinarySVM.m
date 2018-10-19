classdef BinarySVM
    %Encapsulate one binary SVM
    properties
        label     % the one in one-versus-rest
        svlist    % list of support vectors
        b         % intercept
    end
    
    methods
        function obj = BinarySVM(label, svlist, b)
            %Init
            obj.label  = label;
            obj.svlist = svlist;
            obj.b      = b;
        end
    end
end
