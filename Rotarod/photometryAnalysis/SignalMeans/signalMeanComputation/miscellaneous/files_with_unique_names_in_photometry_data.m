addpath(fullfile("..", "utils"));

% dataDir = fullfile(".", "manyData");
dataDir = "Z:\Raymond Lab\2 Colour D1 D2 Photometry Project\YAC128 Mice -- 6 months\Rotarod\Photometry Data";

guessedDepth = guess_data_directory_depth_looking_at_tnt_file(dataDir);
result = list_file_names_found_in_data_directories(dataDir, guessedDepth);
dataKeys = keys(result);

cellArray = {};
for idx = 1:length(dataKeys)
    k = dataKeys(idx);
    cellArray{idx,1} = k{1};

    val = result(k{1});

    for jdx = 1:length(val)
        v = val{jdx};
        cellArray{idx,jdx+1} = v;
    end
    % disp(k{1} + ": " + val(:));
end

cellArray = cellArray';
writecell(cellArray, fullfile("results", "unique_files.csv"));
    


%% HELPER FUNCTIONS
function dataMap = list_file_names_found_in_data_directories(DATA_DIR, n)
    dataMap = containers.Map();

    % as far as depth isn't 1
    if n ~= 1
        % for directories contained in DATA_DIR
        dirs = dir(DATA_DIR);
        for subDir = dirs(3:end)'
            % we check it is indeed a directory
            if subDir.isdir
                subfolderPath = fullfile(subDir.folder, subDir.name);
                additionalDataMap = list_file_names_found_in_data_directories(subfolderPath, ...
                    n-1);
                for key = keys(additionalDataMap)
                    keyContent = key{1};
                    if isKey(dataMap, keyContent)
                        cellEntry = dataMap(keyContent);
                        newEntries = additionalDataMap(keyContent);
                        cellEntry(end + 1:end + length(newEntries)) = newEntries;
                        dataMap(keyContent) = cellEntry;
                    else
                        dataMap(keyContent) = additionalDataMap(keyContent);
                    end
                end
            end
        end
    % otherwise, if we have reached the directory containing the data
    elseif n == 1
        % for directories contained in DATA_DIR
        dirs = dir(DATA_DIR);
        for subDir = dirs(3:end)'
            % ensure our checked one isn't a file but a directory
            if subDir.isdir
                fullpath = fullfile(subDir.folder, subDir.name);
                fileNameList = helper_that_lists_all_file_names_in_directory(fullpath);
                
                for fileName = fileNameList
                    fileNameContent = fileName{1};
                    if isKey(dataMap, fileNameContent)
                        cellEntry = dataMap(fileNameContent);
                        cellEntry{end + 1} = subDir.name;
                        dataMap(fileNameContent) = cellEntry;
                    else
                        dataMap(fileNameContent) = {subDir.name};
                    end
                end
            end
        end
    end

    function returned = helper_that_lists_all_file_names_in_directory(DATADIR)
        returned = {};
        files = dir(DATADIR)';
        skippedFileNames = ["Notes.txt", "StoresListing.txt", "Thumbs.db"];
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
               ~ismember(file.name, skippedFileNames)
                returned{end + 1} = file.name;
            end
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