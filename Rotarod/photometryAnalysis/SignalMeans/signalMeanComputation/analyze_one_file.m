% RUN ANALYSIS ON A SINGLE FILE!

%% Imports
addpath(fullfile("..", "utils"));

%% Useful Constants
% path to data
DATA_DIR = fullfile(...
    "X:\Raymond Lab\2 Colour D1 D2 Photometry Project",...
    "B6-Q175 Mice -- 6 and 10 months",...
    "D1-GCAMP and D2-RCAMP (regular)", ..."D1-RCAMP and D2-GCAMP (switched)",...
    "Rotarod",...
    "Red_Green_Isosbestic-210705\Day2_252_m6_rotarod3" ...
    );
    
    % "Red_Green_Isosbestic-210913\day1_253f3_rotarod3"
%"Red_Green_Isosbestic-220829\day1_327245_f1_rotarod1"
%{
fullfile(...
...
"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\B6-Q175 Mice -- 6 and 10 months\D1-GCAMP and D2-RCAMP (regular)\Rotarod\Photometry Data", ...
"Red_Green_Isosbestic-210909", "day3_242m12_rotarod3"...
);
%}

% path to folder where we save all analysis figures
FIGURE_SAVE_DIR = fullfile(".", "results");
% whether to save figures created as a result of analysis
SAVE_JPG = false;
SAVE_EMF = false;
% whether to display saved figures
DISPLAY_FIGURES = false;
% whether to recalculate entries that are in resultData by name already
RECALCULATE = false;
% whether to show comments detailing how the analysis is going
DISPLAY_COMMENTS = true;

% whether to run the script while skipping errors (as done with
% apply_photometry_analysis_to_many.m) or by stopping at errors (useful for
% debugging)
SKIP_ERROR = false;

%% Function execution
rdFields = Constants_RotarodPhotometry.Means_ResultDataFields;
rd = ResultData(rdFields);

if ~SKIP_ERROR 
    data = photometry_rotarodaccelerating_redandgreen_updatedanalysis( ...
        DATA_DIR, rd, RECALCULATE, SAVE_JPG, SAVE_EMF, ...
        FIGURE_SAVE_DIR, DISPLAY_FIGURES, DISPLAY_COMMENTS);
else
    try
        data = photometry_rotarodaccelerating_redandgreen_updatedanalysis( ...
            DATA_DIR, rd, RECALCULATE, SAVE_JPG, SAVE_EMF, ...
            FIGURE_SAVE_DIR, DISPLAY_FIGURES, DISPLAY_COMMENTS);
    catch ME
        [~,filename,exp] = fileparts(DATA_DIR); filename = append(filename, exp);
        disp("Some error occurred when analyzing : " + ...
             filename + " which is identified as: " + ...
            ME.identifier);
    end
end
