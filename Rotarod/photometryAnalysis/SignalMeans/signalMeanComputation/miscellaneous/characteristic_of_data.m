% VISUALIZE MANY FEATURES (IN NUMBER) OF THE DATASET.
% WE WILL:
% - LOOK AT THE "DATA" STRUCTURE, FINDING THE EPOCS ENTRY
% - FIND OUT HOW MANY VALUES THERE ARE FOR: 1) PtAB.onset, 2) Note.onset

%% VIRTUAL IMPORTS
addpath(fullfile("helpers"));
addpath(fullfile("..", "utils"))

DATA_DIR = fullfile(...
"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\B6-Q175 Mice -- 6 and 10 months\D1-GCAMP and D2-RCAMP (regular)\Rotarod"...
);

SAVE_DIR = fullfile(...
    "results");

FILENAME = "Q175_NumberOfPtABAndNoteEntries.csv";

rdFields = ["Note_onset_size", "PtAB_onset_size"];
rd = ResultData(rdFields);

if isfile(fullfile(SAVE_DIR, FILENAME))
    load_data_for_analysis(rd, SAVE_DIR, FILENAME);
end

run_function_on_photometry_data_found_n_deep( ...
    DATA_DIR, NaN, NaN, false, ... %data_dir, n, filter, stop_on_error
    @get_data_characteristic, rd) % f_handle, varargin

rd.savedata(SAVE_DIR, FILENAME);

%% Analyzes how many entries there are in Notes.onset and PtAB.onset
function get_data_characteristic(dataDir, rd)
    dataDir = convertStringsToChars(dataDir);

    % add path to helpers
    addpath(fullfile(".", "helpers"));
    
    %% Useful constants


    % get and process mouse name
    [~,mouseNameAll,~,~,~] = ...
        process_mouse_name(dataDir);

    %% Import data from TDT files or pre-stored mat file
    % When we have files named "Updated Notes and PtAB.mat", 
    % "Updated Notes.mat", "Updated PrtB and Notes.mat", or 
    % "Updated PtAB.mat" under dataDir, load it. This holds a
    % structure named "data" with (presumably) correct timestamps.
    disp("-------------------------------------------")

    % check if files of the above form are there
    fileNamesToLoad = ["Updated Notes and PtAB.mat", "Updated Notes.mat", ... 
        "Updated PrtB and Notes.mat", "Updated PtAB.mat"];
    checkedDirs = dir(dataDir)';
    % check we only have one file to load with given names; if more, error
    foundOneFileToLoad = false;
    for file = fileNamesToLoad
        if ismember(file, {checkedDirs.name})
            if foundOneFileToLoad
                ME = MException(Constants_RotarodPhotometry.ErrorIdentifierPrefix, ':', ...
                    'moreThanTwoFilesToLoad', ...
                    ['Following folder has more than two files to load updated' ...
                    'timestamps from: %s.'], dataDir);
                throw(ME);
            else 
                foundOneFileToLoad = true;
                fprintf("Filename: %s.\n", dataDir);
                fprintf("Found an updated timestamp version from file: %s - loading!", ...
                    file);
                disp(" ");
            end
            % load data into an object named 'data'
            data = load(fullfile(dataDir, file)); data = data.data;
        end
    end
    % if unable to find any files to load from, load data using TDTbin2mat
    if ~foundOneFileToLoad
        data = TDTbin2mat(dataDir);
    end

    %% Check how many entries there are in Note.onset & PtAB.onset
    note_onset_size = size(data.epocs.Note.onset);
    ptab_onset_size = size(data.epocs.PtAB.onset);

    rd.adddata(mouseNameAll, ...
        struct( ...
        "Note_onset_size", note_onset_size, ...
        "PtAB_onset_size", ptab_onset_size))
end
