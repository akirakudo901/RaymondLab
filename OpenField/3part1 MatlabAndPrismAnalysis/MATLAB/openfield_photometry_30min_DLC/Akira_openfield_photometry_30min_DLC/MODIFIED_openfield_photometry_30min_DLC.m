% clc       % clear content in command line
clear     % clears all variables.
close all % closes all figures


%% Constant declarations
% Declare useful constants.
CLOSE_FIGURES_FOR_EACH_CSV = false;
DISPLAY_FIGURES = false;
% SAVE_FOR_PRISM = false; %THIS IS TEMPORARILY DISABLED
SAVE_ANALYSIS = true;
LOAD_ANALYSIS = false;

% points to directory to which we save all data
% Change to that of your choice
SAVE_DIR = fullfile("data", "results");

MOUSETYPE = ["Q175", "YAC128"];
FILT_OR_UNFILT = ["filt", "unfilt"];

% points to the folder holding all csvs in question
% OR a folder holding folders of csvs divided by mouse type
% Change to that of your choice
MAIN_DATA_DIR = "X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\DLC\%s\csv\allcsv_2024_05_16_Akira";
ESCAPED_MAIN_DATA_DIR = replace(MAIN_DATA_DIR, "\", "\\");

% name of file holding analysis data
COMMON_PREFIX = "WithCenterTimeOverTime_%s_analysis_data_";

% GENERATE THE DATA FOLDERS & CSV NAMES FROM MOUSETYPE AND FILT_OR_UNFILT
DATA_DIRS = {};
ANALYSIS_DATA_CSV_NAMES = {};
for mouseType=MOUSETYPE
    for filt_or_unfilt=FILT_OR_UNFILT
        DATA_DIRS{end+1} = fullfile(sprintf(ESCAPED_MAIN_DATA_DIR, mouseType), ...
            filt_or_unfilt);
        ANALYSIS_DATA_CSV_NAMES{end+1} = strcat(sprintf(COMMON_PREFIX, ...
            mouseType), filt_or_unfilt, '.csv');
    end
end

% path to the folder where prism is located
% should be MATLAB & Prism > openfield...
PRISM_DIR = fullfile("..", "..", "Prism");

%% Function execution
% We apply the analysis function defined below on a set of CSVs.
for i=1:length(DATA_DIRS)
    DATADIR = DATA_DIRS{i};
    ANALYSIS_CSV = ANALYSIS_DATA_CSV_NAMES{i};
    main(DATADIR, ANALYSIS_CSV, SAVE_DIR, PRISM_DIR, DISPLAY_FIGURES, ...
        CLOSE_FIGURES_FOR_EACH_CSV, SAVE_ANALYSIS, LOAD_ANALYSIS)
end

%% Main function
% Executes the analysis.
function main(data_dir, analysis_data_csv_name, save_dir, prism_dir,...
    DISPLAY_FIGURES, CLOSE_FIGURES_FOR_EACH_CSV, SAVE_ANALYSIS, LOAD_ANALYSIS)
    % first check if save_dir / prism_dir is accessible
    if ~isfolder(save_dir)
        ME = MException( ...
            sprintf('%s:noSuchDirectory', Constants_BasicAnalysis.ErrorIdentifierPrefix), ...
            'Saving directory isn''t accessible: %s', save_dir ...
            );
        throw(ME)
    end
    if ~isfolder(prism_dir)
        ME = MException( ...
            sprintf('%s:noSuchDirectory', Constants_BasicAnalysis.ErrorIdentifierPrefix), ...
            'Prism directory isn''t accessible: %s', prism_dir ...
            );
        throw(ME)
    end
    % first check if a csv file stores analysis data already under DATA_DIR
    % if it does, initialize a self-defined ResultData object from it
    analysis_data_map = load_analysis_data_map(analysis_data_csv_name, ...
        save_dir, LOAD_ANALYSIS);
    
    % for every csv files / folders contained under DATA_DIR or its subdirs
    %exclude . and ..
    foldersToCheck = dir(data_dir)'; foldersToCheck = foldersToCheck(3:end);
    for fileObject = foldersToCheck
        % we can have csvs directly
        if ~fileObject.isdir
            analyze_csv_file(fileObject, ...
                data_dir,save_dir,analysis_data_map, ...
                DISPLAY_FIGURES,CLOSE_FIGURES_FOR_EACH_CSV);
    
        % otherwise we might have a subdirectory of name mouse type
        elseif fileObject.isdir
            % check if those hold csvs
            newFullDirPath = fullfile(data_dir, fileObject.name);
            newDirToCheck = dir(newFullDirPath)'; %exclude . and ..
            newDirToCheck = newDirToCheck(3:end);
            % analyze each contained csv
            for csvFile = newDirToCheck
                if endsWith(csvFile.name,".csv")
                    analyze_csv_file(csvFile, ...
                        newFullDirPath,save_dir,analysis_data_map, ...
                        DISPLAY_FIGURES,CLOSE_FIGURES_FOR_EACH_CSV);
                end
            end
            ...
        end
        ...
    end
    
    % finally, save the analysis data we just generated
    save_analysis_data_map(analysis_data_map, ...
        analysis_data_csv_name, save_dir, prism_dir, SAVE_ANALYSIS);
end

%% Helper functions
% Checks if a file titled ANALYSIS_DATA_CSV_NAME exists under SAVE_DIR
% If yes, return a ResultData object by reading from it, which maps
%  file names to their data (to avoid overlap)
% Otherwise, initialize an empty ResultData object
function dataMap = load_analysis_data_map(fileName, saveDir, LOAD_ANALYSIS)
    % initialize an empty ResultData with specific field
    dataMap = ResultData(Constants_BasicAnalysis.BasicAnalysis_ResultDataFields);
    fullFilePath = fullfile(saveDir, fileName);
    % if the file in question exists, load its content into new ResultData
    if exist(fullFilePath, "file") && LOAD_ANALYSIS
        dataMap.loaddata(fullFilePath);
    end
end

% Saves a containers.Map object into csv with name ANALYSIS_DATA_CSV_NAME
% under directory SAVE_DIR.
% Map object is converted to an equivalent cell array before saving in csv.
function save_analysis_data_map(savedMap, fileName, saveDir, prismDir, ...
    SAVE_ANALYSIS)
    if ~SAVE_ANALYSIS 
        return
    end
    % assuming passed savedMap is of type ResultData, simply run its save
    savedMap.savedata(saveDir, fileName);
    % also record data for prism
%     if SAVE_FOR_PRISM
%         commonName = erase(fileName, ".csv");
%         savedMap.savedataforprism(prismDir, commonName);
%     end
end

% Run an analysis on a csv file
function analyze_csv_file(csvFile, ...
    dataDir,saveDir,resultData,DISPLAY_FIGURES,CLOSE_FIGURES_FOR_EACH_CSV)
    % if it is a csv file, we want to analyze it
    csvFileName = csvFile.name; 
    if endsWith(csvFileName, '.csv')

        disp("Processing file " + csvFileName + "...")

        % create a regular expression to extract mouse name from file name
        % this is of the form: '6 digits', '_', 'm'/'f', '1/2 digits'
        expression = '\d{6}_?[mf]\d{1,2}';
        mouseName = regexp(csvFileName,expression,'match');
        disp(append("Mouse name is identified as: ", mouseName, "!"));
        
        % run the analysis
        csvPath = fullfile(dataDir, csvFileName); % path to csv
        % passing mouseName as graph title
        % we also catch error when user decides not to proceed given a
        %  video shorter than desired
        try
            analysis(csvPath,saveDir,mouseName,resultData,true,DISPLAY_FIGURES);
            disp("File " + csvFileName + " processed successfully!")
            disp("")
            % if CLOSE_FIGURES_FOR_EACH_CSV is true, close open figures
            if CLOSE_FIGURES_FOR_EACH_CSV
                close all
            end
        % catch any Matlab Exception
        catch ME
            % if the exception isn't the video being too short, rethrow
            if ~strcmp(ME.identifier,'OpenfieldPhotometry30min:videoIsTooShort')
                rethrow(ME);
            end
        end
    end
    ...
end