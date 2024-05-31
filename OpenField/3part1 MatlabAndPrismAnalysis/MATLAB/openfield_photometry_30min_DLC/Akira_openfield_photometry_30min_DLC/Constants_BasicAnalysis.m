%% Author: Akira Kudo
% This file defines constants that are useful to share across files
% in the basic analysis for DLC results.

classdef Constants_BasicAnalysis
    properties( Constant = true )
        ErrorIdentifierPrefix = 'BasicAnalysis'
        BasicAnalysis_ResultDataFields = [
            "mouseType","totalDistanceCm","centerTime","centerTimeByIntervals", ...
            "timeFractionByQuadrant", "distanceByIntervals"
            ]
        
    end
end