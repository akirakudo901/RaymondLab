%% Apply photometry analysis to many
% Applies photometry analysis to a set of data contained in many folders
% populating a single given folder.
% All data from analysis is then saved into one csv and stored at the
% specified path.

%% Virtual Imports
% importing ResultData
addpath(fullfile("..", "utils"), ...
        fullfile("helpers"));

%% Useful constants
% path to directory holding all data directories we analyze
DATA_DIRS = [...
    "X:\Raymond Lab\2 Colour D1 D2 Photometry Project\B6-Q175 Mice -- 6 and 10 months\D1-GCAMP and D2-RCAMP (regular)\Rotarod\Red_Green_Isosbestic-210705\Day2_252_m6_rotarod3", ...
    "X:\Raymond Lab\2 Colour D1 D2 Photometry Project\B6-Q175 Mice -- 6 and 10 months\D1-GCAMP and D2-RCAMP (regular)\Rotarod\Red_Green_Isosbestic-220908\day1_349412m7_rotarod1" ...<- day1
    ];    
% 
% "X:\Raymond Lab\2 Colour D1 D2 Photometry Project\B6-Q175 Mice -- 6 and 10 months\D1-GCAMP and D2-RCAMP (regular)\Rotarod\Red_Green_Isosbestic-220910\day1_349412m7_rotarod1", ... <- day3

% Q175
%"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\B6-Q175 Mice -- 6 and 10 months\D1-GCAMP and D2-RCAMP (regular)\Rotarod",...
%"X:\Raymond Lab\2 Colour D1 D2 Photometry Project\B6-Q175 Mice -- 6 and 10 months\D1-RCAMP and D2-GCAMP (switched)\Rotarod"

% YAC128
% "Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\YAC128 Mice -- 6 months\Rotarod\Photometry Data";

% path + name to csv we save analysis result into
CSV_SAVE_DIR = fullfile(".", "results");
% CSV_NAME = "ALL_BUT_SPECIAL_FILES_photometry_means_analysis_result.csv";
ANALYSIS_PREFIX = "Q175AdditionalInfo";
CSV_NAME = ANALYSIS_PREFIX + "_photometry_means_analysis_result.csv";

% path to folder where we save all analysis figures
FIGURE_SAVE_DIR = fullfile(".", "results", ANALYSIS_PREFIX);
% whether to save figures created as a result of analysis
SAVE_JPG = true;
SAVE_EMF = true;
% whether to display saved figures
DISPLAY_FIGURES = false;
% whether to recalculate entries that are in resultData by name already
RECALCULATE = false;
% whether to print comments indicating status
DISPLAY_COMMENTS = true;


% whether to keep going if we get an error from processing one file
STOP_UPON_ERROR = false;
% whether to load data from csv
LOAD_DATA = true;
% whether to save data to csv
SAVE_DATA = true;
% whether to fill empty trials with dummy ones
FILL_EMPTY_TRIALS = true;

% whether to keep console output to a txt file, recommended is True!
KEEP_DIARY = true;
% file to which the diary will be saved
DIARYNAME = ANALYSIS_PREFIX + "_signalMeans_computation_log.txt";
DIARY_PATH = fullfile(...
    "X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Rotarod\photometryAnalysis\SignalMeans\signalMeanComputation\results", ...
    ANALYSIS_PREFIX, DIARYNAME);


%% Sanity checks for helper functions doing the correct thing
sanity_check_entryIsNotContainedInThisFile()


%% Function execution
if ~isfile(FIGURE_SAVE_DIR)
    disp("Directory to which we want to save figures doesn't exist - creating:")
    disp(FIGURE_SAVE_DIR + "!")
    mkdir(FIGURE_SAVE_DIR)
end

if KEEP_DIARY
    disp("Saving log into " + DIARYNAME + " located at " + DIARY_PATH)
    diary(DIARY_PATH)
end

rdFields = Constants_RotarodPhotometry.Means_ResultDataFields;
rd = ResultData(rdFields);

% first load potentially existing data
if LOAD_DATA
    load_data_for_analysis(rd, CSV_SAVE_DIR, CSV_NAME);
end

% Define a filter
notInFile = entryIsNotContainedInThisFile( ...
    fullfile("X:\Raymond Lab\2 Colour D1 D2 Photometry Project\Akira\Rotarod\photometryAnalysis\SignalMeans\signalMeanComputation\results", ...
             CSV_NAME), ...
    rdFields, ["Means_Green", "Meanshift_Green", "Means_Red", "Meanshift_Red"]);
% "C:\Users\mashi\Desktop\RaymondLab\Experiments\Rotarod\photometryAnalysis\SignalMeans\Akira_photometry_rotarod_scripts\Ellen script\results\ALL_BUT_SPECIAL_FILES_photometry_means_analysis_result.csv", ...
    

% doesNotContainNames = @(dataDir) ...
%     ~contains(dataDir, "day4_395034_f5_rotarod2") && ...
%     ~contains(dataDir, "day3_372301_m3_rotarod2");

containsNames = @(dataDir) ...
    contains(lower(dataDir), ...
    [ ...
    "day3_242m12_rotarod3",    "day3_242_m12_rotarod3", ...%9
    "day4_248m23_rotarod3",    "day4_248_m23_rotarod3",...%12
    "day3_312142f3_rotarod1",  "day3_312142_f3_rotarod1", ...%7
    "day1_312144m23_rotarod1", "day1_312144_m23_rotarod1",...%1
    "day2_320151m1_rotarod1",  "day2_320151_m1_rotarod1",...%4
    "day1_327245f1_rotarod1",  "day1_327245_f1_rotarod1", ...%1
    "day3_349412m7_rotarod1",  "day3_349412_m7_rotarod1", ...%7
    "day3_361212f5_rotarod2",  "day3_361212_f5_rotarod2", ...%8
    "day2_361214m3_rotarod2",  "day2_361214_m3_rotarod2",...%5
    "day1_363451f5_rotarod2",  "day1_363451_f5_rotarod2",...%2
    "day3_372301m3_rotarod2",  "day3_372301_m3_rotarod2",...%8
    "day1_524f1_rotarod1",     "day1_524_f1_rotarod1",... %1
    "day1_524f1_rotarod2",     "day1_524_f1_rotarod2",... %2
    "day2_524f1_rotarod1",     "day2_524_f1_rotarod1",... %4
    "day1_524f23_rotarod3",    "day1_524_f23_rotarod3",...%3
    "day4_525f2_rotarod1",     "day4_525_f2_rotarod1", ...%10
    "day4_829f1_rotarod1",     "day4_829_f1_rotarod1",...%10
    "day4_328147m1_rotarod3",  "day4_328147_m1_rotarod3",...%12
    "day1_524f1_rotarod3",     "day1_524_f1_rotarod3" ...%3_2
    ]);

%    contains(erase(dataDir, ["-", "_"]), ...
%    ["242m12", "248m23", "312142f3", "312144m23", "320151m1", ...
%     "327245f1", "349412m7", "361212f5", "361214m3", "363451f5", ...
%     "372301m3", "392607m5", "524f1", "524f23", "525f2", "829f1"...
%     "328147m1", "524f1"]);

combined = NaN; %@(dataDir) (containsNames(dataDir) && notInFile(dataDir));

% notInFileAndNoSpecialFiles = @(dataDir) (notInFile(dataDir) && ...
%     doesNotContainAnySpecialFiles(dataDir));

rd = main_computation(DATA_DIRS, rd, combined, RECALCULATE, ...
    SAVE_JPG, SAVE_EMF, FIGURE_SAVE_DIR, DISPLAY_FIGURES, STOP_UPON_ERROR);

% finally save data
if SAVE_DATA
    save_data_for_analysis(rd, CSV_SAVE_DIR, CSV_NAME);
end

% also consider filling emmpty trial entries to format data for Prism
if FILL_EMPTY_TRIALS && SAVE_DATA
    fill_empty_trial_entries(fullfile(CSV_SAVE_DIR, CSV_NAME));
end

if KEEP_DIARY
    diary off
end


%% Helper functions
% Function for applying the main computation two multiple data directories
function rd = main_computation(data_directories, rd, filter, RECALCULATE, ...
    SAVE_JPG, SAVE_EMF, FIGURE_SAVE_DIR, DISPLAY_FIGURES, STOP_UPON_ERROR)
    for data_dir = data_directories
        % run the analysis, specifying depth of data relative to DATA_DIR
        % we can use a heuristic guess_data_directory... to guess the depth
        guessedDepth = guess_data_directory_depth_looking_at_tnt_file(data_dir);
        disp("guessedDepth is: " + guessedDepth);
    
        warning('off', 'MATLAB:polyfit:RepeatedPointsOrRescale');
        
        try
            analyze_data_found_N_deep(data_dir, guessedDepth, rd, ...
                filter, RECALCULATE, SAVE_JPG, SAVE_EMF, ...
                FIGURE_SAVE_DIR, DISPLAY_FIGURES, STOP_UPON_ERROR);
        catch ME
            disp("Caught the following error:")
            disp(ME.message);
            disp(ME.stack);
        end
    end
end

% Function for any loading acts
function load_data_for_analysis(rd, CSV_SAVE_DIR, CSV_NAME)
    if isfile(fullfile(CSV_SAVE_DIR, CSV_NAME))
        rd.loaddata(fullfile(CSV_SAVE_DIR, CSV_NAME));
    end
end

% Function for any saving acts
function save_data_for_analysis(rd, CSV_SAVE_DIR, CSV_NAME)
    rd.savedata(CSV_SAVE_DIR, CSV_NAME);
end

% Allows you to specify how deep data directories (which hold the
% miscellaneous files read for analysis) are relative to the directory
% given as DATA_DIR. 
% For example, if we have 12 folders holding data which each belong to one 
% of four folders, which are in turn under DATA_DIR, the depth is 2.

% doYouRunAnalysis : Function handle that allows you to determine whether
% to run the analysis, when given full path to the directory we are
% scanning for data. 
% - If NaN is passed, runs analysis on every entry
% EXAMPLE USE: only analyze if folder name indicates this is a male
function analyze_data_found_N_deep(DATA_DIR, n, rd, doYouRunAnalysis, ...
    RECALCULATE, SAVE_JPG, SAVE_EMF, FIGURE_SAVE_DIR, DISPLAY_FIGURES, STOP_UPON_ERROR)
    % as far as depth isn't 0
    if n > 0
        % for directories contained in DATA_DIR
        dirs = dir(DATA_DIR);
        for subDir = dirs(3:end)'
            % we check it is indeed a directory
            if subDir.isdir
                subfolderPath = fullfile(subDir.folder, subDir.name);
                analyze_data_found_N_deep(subfolderPath, n-1, rd, ...
                    doYouRunAnalysis, RECALCULATE, SAVE_JPG, SAVE_EMF, ...
                    FIGURE_SAVE_DIR, DISPLAY_FIGURES, STOP_UPON_ERROR);
            end
        end
    % otherwise, if we have reached the directory containing the data
    elseif n == 0
        % custom check based on value of DATA_DIR to determine if
        % we run the analysis on this folder
        if ((~isa(doYouRunAnalysis,'function_handle') && ...
              isnan(doYouRunAnalysis)) || (doYouRunAnalysis(DATA_DIR)))
            if ~STOP_UPON_ERROR
                try
                    photometry_rotarodaccelerating_redandgreen_updatedanalysis( ...
                        DATA_DIR, rd, RECALCULATE, SAVE_JPG, SAVE_EMF, ...
                        FIGURE_SAVE_DIR, DISPLAY_FIGURES, true);
                catch ME
                    disp("Some error occurred when analyzing : " + ...
                        DATA_DIR + " which is identified as: " + ...
                        ME.identifier);
                end
            else
                photometry_rotarodaccelerating_redandgreen_updatedanalysis( ...
                        DATA_DIR, rd, RECALCULATE, SAVE_JPG, SAVE_EMF, ...
                        FIGURE_SAVE_DIR, DISPLAY_FIGURES, true);
            end
        ...
        end
    end 
end

% A heuristic guess of the depth of data directories relative to the
% directory given as DATA_DIR - returns the first depth at which a .tnt 
% file exists, which is the extension of data files
% If we don't find a .tnt file anywhere, we return -1.
% *THIS FUNCTION ASSUMES THAT ALL DATA DIRECTORIES ARE AT AN EQUAL DEPTH,
% AND THAT A DATA DIRECTORY DOES NOT CONTAIN A DIRECTORY ITSELF - HENCE 
% PERFORMING DEPTH FIRST SEARCH
function depth = guess_data_directory_depth_looking_at_tnt_file(DATA_DIR)
    disp("==============================================");
    disp("guess_data_directory_depth_looking_at_tnt_file assumes that:");
    disp("1) no data directory hold folders");
    disp("2) all data directories are found at the same depth!");
    disp("==============================================");
    depth = inner_search(DATA_DIR, 0);

    function depth = inner_search(checkedDir, depthSoFar)
        depth = -1; % default for not finding anything
        % for directories contained in dir
        dirs = dir(checkedDir);
        for subDir = dirs(3:end)'
            % we check it is indeed a directory
            if subDir.isdir
                subfolderPath = fullfile(subDir.folder, subDir.name);
                depth = inner_search(subfolderPath, depthSoFar + 1);
                % if we found a data directory (e.g. depth != -1), return
                if depth ~= -1 return; end
            % if we see a non-directory element, check if it ends with tnt
            else
                if endsWith(subDir.name, '.tnt')
                    depth = depthSoFar;
                    return;
                end
            end
        end
    end
end

% HELPERS FOR WHETHER TO RUN THE ANALYSIS
% Returns true if there are no "special files", as defined in the function
% below, contained in this data directory.
function noSpecialFiles = doesNotContainAnySpecialFiles(dataDir)
    skippedFileNames = ["Notes.txt", "StoresListing.txt", "Thumbs.db", ...
        "matlab.mat", "Rotarod_Frames.mat", "Rotarod_Times.mat", ...
        "Timestamps.mat", "Akira_memo.txt", "Changes.json"];
    
    noSpecialFiles = true;

    files = dir(dataDir)';
    for file = files(3:end)
        if ~endsWith(file.name, '.emf') && ...
           ~endsWith(file.name, '.jpg') && ...
           ~endsWith(file.name, '.avi') && ...
           ~endsWith(file.name, '.mp4') && ...
           ~endsWith(file.name, '.Tbk') && ...
           ~endsWith(file.name, '.Tdx') && ...
           ~endsWith(file.name, '.tev') && ...
           ~endsWith(file.name, '.tin') && ...
           ~endsWith(file.name, '.tnt') && ...
           ~endsWith(file.name, '.tsq') && ...
           ~endsWith(file.name, '.csv') && ...
           ~endsWith(file.name, '.h5') && ...
           ~endsWith(file.name, '.pickle') && ...
           ~endsWith(file.name, '.ini') && ...
           ~endsWith(file.name, '.fig') && ...
           ~endsWith(file.name, 'trimmed.mat') && ...
           ~ismember(file.name, skippedFileNames)
            noSpecialFiles = false;
            return
        end
    end
end

% fieldsEmptyTobeConsideredNotContained : if those fields are empty
% for an entry (in our case expressed as empty char array ''), we consider 
% this entry as not contained in this file
function handle = entryIsNotContainedInThisFile(filePath, rdFields, ...
    fieldsEmptyTobeConsideredNotContained)
    if ~isfile(filePath)
        disp("filePath didn't exist; always returning true:");
        disp(filePath);
        handle = @(datadir) true;
        return
    end
    rdInternal = ResultData(rdFields);
    rdInternal.loaddata(filePath);

    handle = @fileIsNotContainedIn;

    function notContained = fileIsNotContainedIn(dataDir)
        [~,folderName,~] = fileparts(dataDir);
        
        [~,mouseNameTotalTrial,~,~,~] = process_mouse_name(folderName);

        if ~rdInternal.iskey(mouseNameTotalTrial)
            notContained = true;
            return
        else
            notContained = true;
            for field = fieldsEmptyTobeConsideredNotContained
                if string(rdInternal.get(mouseNameTotalTrial).(field)) ~= ""
                    notContained = false; 
                    return
                end
            end
        end
    end

end

function sanity_check_entryIsNotContainedInThisFile()
    disp("=====Sanity check entryIsNotContainedInThisFile start!=====")
    filePath = fullfile("manyData", ...
        "test_these_should_be_excluded.csv");
    rdFields = ["Day", "Trial"];
    notInFile = entryIsNotContainedInThisFile(filePath, rdFields, ...
        ["Day", "Trial"]);
    
    fileNames = ["day1_12345_m1_rotarod1", "day1_12345_m1_rotarod2", ...
                 "day1_12345_m1_rotarod3", "day2_12345_m1_rotarod1"];
    isExpectedToNotBeInFile = [false, true, true, false];
    for idx = 1:length(fileNames)
        fn = fileNames(idx);
        assert(notInFile(fn) == isExpectedToNotBeInFile(idx));
    end
    disp("=====Sanity check entryIsNotContainedInThisFile end!=====")
end