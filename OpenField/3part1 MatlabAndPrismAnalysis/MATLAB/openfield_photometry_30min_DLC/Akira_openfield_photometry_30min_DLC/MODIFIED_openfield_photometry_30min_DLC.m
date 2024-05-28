% clc       % clear content in command line
clear     % clears all variables.
close all % closes all figures


%% Constant declarations
% Declare useful constants.
CLOSE_FIGURES_FOR_EACH_CSV = false;
DISPLAY_FIGURES = false;

% points to directory to which we save all data
% Change to that of your choice
SAVE_DIR = fullfile("data", "results");

% points to the folder holding all csvs in question
% OR a folder holding folders of csvs divided by mouse type
% Change to that of your choice

DATA_DIR = fullfile("data", "csvs", "Q175", "trunc", "unfilt");


% name of file holding analysis data
ANALYSIS_DATA_CSV_NAME = "Q175_analysis_data_unfilt.csv";

% path to the folder where prism is located
% should be MATLAB & Prism > openfield...
PRISM_DIR = fullfile("..", "Prism_temp");



%% Function execution
% We apply the analysis function defined below on a set of CSVs.

% first check if a csv file stores analysis data already under DATA_DIR
% if it does, initialize a self-defined ResultData object from it
analysis_data_map = load_analysis_data_map(ANALYSIS_DATA_CSV_NAME, SAVE_DIR);

% for every csv files / folders contained under DATA_DIR or its subdirs
%exclude . and ..
foldersToCheck = dir(DATA_DIR)'; foldersToCheck = foldersToCheck(3:end);
for fileObject = foldersToCheck
    % we can have csvs directly
    if ~fileObject.isdir
        analyze_csv_file(fileObject, ...
            DATA_DIR,SAVE_DIR,analysis_data_map, ...
            DISPLAY_FIGURES,CLOSE_FIGURES_FOR_EACH_CSV);

    % otherwise we might have a subdirectory of name mouse type
    elseif fileObject.isdir
        % check if those hold csvs
        newFullDirPath = fullfile(DATA_DIR, fileObject.name);
        newDirToCheck = dir(newFullDirPath)'; %exclude . and ..
        newDirToCheck = newDirToCheck(3:end);
        % analyze each contained csv
        for csvFile = dir(newFullDirPath)'
            if endsWith(csvFile.name,".csv")
                analyze_csv_file(csvFile, ...
                    newFullDirPath,SAVE_DIR,analysis_data_map, ...
                    DISPLAY_FIGURES,CLOSE_FIGURES_FOR_EACH_CSV);
            end
        end
        ...
    end
    ...
end

% finally, save the analysis data we just generated
save_analysis_data_map(analysis_data_map, ...
    ANALYSIS_DATA_CSV_NAME, SAVE_DIR, PRISM_DIR);


%% Helper functions
% Checks if a file titled ANALYSIS_DATA_CSV_NAME exists under SAVE_DIR
% If yes, return a ResultData object by reading from it, which maps
%  file names to their data (to avoid overlap)
% Otherwise, initialize an empty ResultData object
function dataMap = load_analysis_data_map(fileName, saveDir)
    % initialize an empty ResultData
    dataMap = ResultData;
    fullFilePath = fullfile(saveDir, fileName);
    % if the file in question exists, load its content into new ResultData
    if exist(fullFilePath, "file")
        dataMap.loaddata(fullFilePath);
    end
end

% Saves a containers.Map object into csv with name ANALYSIS_DATA_CSV_NAME
% under directory SAVE_DIR.
% Map object is converted to an equivalent cell array before saving in csv.
function save_analysis_data_map(savedMap, fileName, saveDir, prismDir)
    % assuming passed savedMap is of type ResultData, simply run its save
    savedMap.savedata(saveDir, fileName);
    % also record data for prism
    commonName = erase(fileName, ".csv");
    savedMap.savedataforprism(prismDir, commonName);
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