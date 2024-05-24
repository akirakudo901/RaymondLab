%% Author: Akira Kudo
% This file defines constants that are useful to share across files
% utilizing the rotarod photometry ensemble.

classdef Constants_RotarodPhotometry
    properties( Constant = true )
        ErrorIdentifierPrefix = 'RotarodPhotometry'
        Means_ResultDataFields = [
            "Day", "Trial", "Means_Green", "Meanshift_Green", "Means_Red", ...
            "Meanshift_Red", "MouseName", "TotalTrial", "After_TooShort", ...
            "Note_Onset_Size1", "Note_Onset_Size2", "PtAB_Onset_Size1", ...
            "PtAB_Onset_Size2", "Exception", "Info"...
            ]
    end
end