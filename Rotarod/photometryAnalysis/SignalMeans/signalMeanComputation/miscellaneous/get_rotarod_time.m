% Author: Akira Kudo
%% Get Rotarod Time
% Will calculate time of mouse on rotarod using data.epocs.PtAB.onset .

%% VIRTUAL IMPORTS
addpath(fullfile("helpers"));
addpath(fullfile("..", "utils"))

%% USEFUL CONSTANTS
DATA_DIR = fullfile(...
"Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\YAC128 Mice -- 6 months\Rotarod\Photometry Data" ...
);
% "Red_Green_Isosbestic-210807"...%, "day2_291475_m3_rotarod1" ...


SAVE_DIR = fullfile(...
    "results");

FILENAME = "Rotarod_LatencyToFall.csv";

RECALCULATE = false; 
DISPLAY_COMMENTS = true;

%% EXECUTION

rdFields = ["MouseName", "TotalTrial", "LatencyToFall", "NumEntryPtAB", ...
            "TimeSpentOutOfCage", "NumEntryNote"];
rd = ResultData(rdFields);

run_function_on_photometry_data_found_n_deep( ...
    DATA_DIR, NaN, NaN, true, ... %data_dir, n, filter, stop_on_error
    @get_rotarod_time_of_given_file, ... % f_handle
    rd, RECALCULATE, DISPLAY_COMMENTS) %varargin

rd.savedata(SAVE_DIR, FILENAME);

function get_rotarod_time_of_given_file(dataDir, ...
    resultData, recalculate, DISPLAY_COMMENTS)
    dataDir = convertStringsToChars(dataDir);

    % add path to helpers
    addpath(fullfile(".", "helpers"));
    
    %% Useful constants
    % files with these names hold data with updated time stamps so we load 
    % them instead of other data
    FILES_TO_LOAD_IF_EXISTING_IN_FOLDER = [
        "Updated Notes and PtAB.mat", "Updated Notes.mat", ... 
        "Updated PrtB and Notes.mat", "Updated PtAB.mat"];


    %% Execution
    % get and process mouse name
    [mouseName,mouseNameAll,~,~,totalTrial] = ...
        process_mouse_name(dataDir);

    % if an entry with the same data is in resultData and parameter 
    % 'recalculate' is false, skip this calculation
    if ~recalculate && resultData.iskey(mouseNameAll)
        if DISPLAY_COMMENTS
            disp("Entry with same name found in resultData, skipping!");
        end
        return
    end

    %% Import data from TDT files or pre-stored mat file
    % When we have files named "Updated Notes and PtAB.mat", 
    % "Updated Notes.mat", "Updated PrtB and Notes.mat", or 
    % "Updated PtAB.mat" under dataDir, load it. This holds a
    % structure named "data" with (presumably) correct timestamps.
    if DISPLAY_COMMENTS
        disp("-------------------------------------------")
    end

    % check if files of the above form are there
    checkedDirs = dir(dataDir)';
    % check we only have one file to load with given names; if more, error
    foundOneFileToLoad = false;
    for file = FILES_TO_LOAD_IF_EXISTING_IN_FOLDER
        if ismember(file, {checkedDirs.name})
            if foundOneFileToLoad
                ME = MException(Constants.ErrorIdentifierPrefix, ':', ...
                    'moreThanTwoFilesToLoad', ...
                    ['Following folder has more than two files to load updated' ...
                    'timestamps from: %s.'], dataDir);
                throw(ME);
            else 
                foundOneFileToLoad = true;
                if DISPLAY_COMMENTS
                    fprintf("Filename: %s.\n", dataDir);
                    fprintf("Found an updated timestamp version from file: %s - loading!\n", ...
                        file);
                end
            end
            % load data into an object named 'data'
            data = load(fullfile(dataDir, file)); data = data.data;
        end
    end
    % if unable to find any files to load from, load data using TDTbin2mat
    if ~foundOneFileToLoad
        data = TDTbin2mat(dataDir, 'TYPE', {'epocs'});
    end

    %% Execution
    % Check that a PtAB entry exists
    try
        data.epocs.PtAB.onset(1,1);
    catch ME
        if strcmp(ME.message, "Unrecognized field name ""PtAB"".")
            if DISPLAY_COMMENTS
                disp("This file apparently has no 'PtAB' entry - " + ...
                    "likely a failed attempt. Skipping.");
            end
            % save an empty result into resultData
            newData = struct( ...
                "MouseName", mouseName, ...
                "TotalTrial", totalTrial, ...
                "LatencyToFall", '', ...
                "NumEntryPtAB", '' ...
                );

            resultData.adddata(append("Failed_", mouseNameAll), ...
                               newData);
            return
        end
    end

    % otherwise, calculate latency based on the first and last PtAB entry
    % also calculate note onsets, may be useful
    latencyToFall = data.epocs.PtAB.onset(end,1) - data.epocs.PtAB.onset(1,1);
    [numEntryPtAB, ~] = size(data.epocs.PtAB.onset);

    timeSpentOutOfCage = data.epocs.Note.onset(end,1) - data.epocs.Note.onset(1,1);
    [numEntryNote, ~] = size(data.epocs.Note.onset);

    % save it into resultData
    newData = struct( ...
        "MouseName", mouseName, ...
        "TotalTrial", totalTrial, ...
        "LatencyToFall", latencyToFall, ...
        "NumEntryPtAB", numEntryPtAB, ...
        "TimeSpentOutOfCage", timeSpentOutOfCage, ...
        "NumEntryNote", numEntryNote ...
        );

    resultData.adddata(mouseNameAll, newData);

end