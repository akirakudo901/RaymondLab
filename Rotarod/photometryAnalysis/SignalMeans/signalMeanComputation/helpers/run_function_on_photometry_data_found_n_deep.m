%% Author: Akira Kudo

% Allows you to run a given function specify how deep data directories (which hold the
% miscellaneous files read for analysis) are relative to the directory
% given as DATA_DIR. 

% data_dir : Directory from which we start searching the data, to N deep.
% n : Depth of search for data. E.g. 12 folders holding data, each 
%     belonging to 1 of 4 folders, in turn under data_dir => depth is 2.
%     If NaN is passed, we use the
%     guess_data_directory_depth_looking_at_tnt_file function.
% filter : Function handle that allows you to determine whether
%          to run the analysis, when given full path to the directory we 
%          are scanning for data. 
%         - If NaN is passed, runs analysis on every entry
%         EXAMPLE USE: only analyze if folder name indicates a male
% stop_on_error : Whether to stop or continue execution while encountering 
%                 an error.
% f_handle : Function handle which is ran on the photometry data, takes in 
%            the path to a data directory. Use varargin for options.
% varargin : Optional arguments passed to f_handle.
function run_function_on_photometry_data_found_n_deep( ...
    data_dir, n, filter, stop_on_error, f_handle, varargin)
    
    % if NaN is passed, we guess the depth
    if isnan(n)
        n = guess_data_directory_depth_looking_at_tnt_file(data_dir);
        if n == -1
            disp("We couldn't guess the depth correctly...")
            return
        end
    % if depth is smaller than -1, raise error
    elseif n < -1
        depthTooSmallException = MException( ...
            append(Constants_RotarodPhotometry.ErrorIdentifierPrefix, ":", ...
            "depthTooSmall") ...
        );
        throw(depthTooSmallException);
    end

    % as far as depth is greater than 1
    if n > 1
        % for directories contained in DATA_DIR
        dirs = dir(data_dir);
        for subDir = dirs(3:end)'
            % we check it is indeed a directory
            if subDir.isdir
                subfolderPath = fullfile(subDir.folder, subDir.name);
                run_function_on_photometry_data_found_n_deep(subfolderPath, ...
                    n-1, filter, stop_on_error, f_handle, varargin{:});
            end
        end
    % otherwise, if we have reached the directory containing the data
    elseif n == 1
        % for directories contained in DATA_DIR
        dirs = dir(data_dir);
        for subDir = dirs(3:end)'
            % ensure our checked one isn't a file but a directory
            if subDir.isdir
                fullpath = fullfile(subDir.folder, subDir.name);
                % custom check based on value of fullpath to determine if
                % we run the analysis on this folder
                if ((~isa(filter,'function_handle') && ...
                      isnan(filter)) || (filter(fullpath)))
                    if ~stop_on_error
                        try
                            f_handle(fullpath, varargin{:});
                        catch ME
                            disp("Some error occurred when analyzing : " + ...
                                subDir.name + " which is identified as: " + ...
                                ME.identifier);
                        end
                    else
                        f_handle(fullpath, varargin{:});
                    end
                ...
                end
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