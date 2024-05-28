%% Checking results produced from script match expected ones
% Run the result_is_computed_correctly function to ensure result is correct    

% adding paths to the result-obtaining codes and their directories first
addpath(fullfile('Akira_openfield_photometry_30min_DLC'));
addpath(fullfile('test'));

% path to the directory holding the csvs we are using for testing
dataDir = fullfile('test');
% this is where the data created by MODIFIED_openfield... is stored
saveDir = fullfile('test', 'demo_target_of_file_creation');

% two csvs to check
csvPath315955m1 = "315955_m1DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv";
csvPath326787m2 = "20220213051553_326787_m2DLC_resnet50_WhiteMice_OpenfieldJan19shuffle1_1030000.csv";

% for each, we run this function
for csvFolder = ["315955_m1",     "326787_m2";
                 csvPath315955m1, csvPath326787m2]
    folderName = csvFolder(1);
    csvName = csvFolder(2);
    analysis(fullfile(dataDir, csvName), ...
             fullfile(saveDir, folderName), ...
             folderName, ...
             ResultData, ...
             true, ...
             false);
end

% run it for every dataId in the list
% dataId should match folder name under data
dataIdArray = ["315955_m1", "326787_m2"];

for dataId = dataIdArray
    assert_result_is_computed_correctly(dataId);
end


%% Function that passes appropriate paths to result_is_computed_correctly
function assert_result_is_computed_correctly(dataId)
    
    % these hold master data for reference
    EXPECTED_DATA_PATH = fullfile('test');
    % these hold actual data created by the "MODIFIED_openfield..." script
    ACTUAL_DATA_PATH = fullfile('test','demo_target_of_file_creation');

    % generate paths to expected & actual directory structures from dataId
    expected315955m1 = convertStringsToChars( ...
        fullfile(EXPECTED_DATA_PATH, dataId));
    actual315955m1 = convertStringsToChars( ...
        fullfile(ACTUAL_DATA_PATH, dataId));
    
    % check if computed correctly
    [isCorrect,diff] = result_is_computed_correctly( ...
        expected315955m1, actual315955m1);
    
    % make it into an assertion
    assert(isCorrect, ['Computation resulted in difference:', ...
        convertStringsToChars(diff)]);
    disp("Test for " + dataId + " passed!");

end

%% Function result_is_computed_correctly 
% This is a comparator function, which compares and check that the 
% directory structure within path is identical to the files in the 
% "expected directory structure".

% An easy sanity check that my code does exactly what it's supposed to be
% doing.

function [isCorrect, diff] = result_is_computed_correctly( ...
    referencePath, resultPath)

    % construct the command - we call WinGNU32 diff which can be found
    % here [https://gnuwin32.sourceforge.net/packages/diffutils.htm]
    diffPath = fullfile('test','lib','diff.exe');
    % diffPath = '"C:\Program Files (x86)\GnuWin32\bin\diff.exe"';
    command = [diffPath ' -r --text ' ...                  % command
               '"' referencePath '" ' ...                  % reference
               '"' convertStringsToChars(resultPath) '"']; % difference

    % run the command
    [status, cmdout] = system(command);
    
    % status is: 0- no difference, 1- difference found, 2- error
    if status == 2
        disp("An error occurred - double check path!"); disp(cmdout);
        isCorrect = false; diff = "";
        return
    elseif status == 0
        isCorrect = true; diff = "";
        return
    else
        isCorrect = false; diff = cmdout;
    end
end


% WHAT I USED TO CALL FOR COMMAND:
% compareDirectoriesPath = [' \"C:\Users\mashi\Desktop\RaymondLab' ...
%         '\MATLAB\open field photometry script\' ...
%         'powershellScript\Compare-Directories.ps1\" '];
% % construct the command - we call a self-made command
% % "Compare-Directories($Dir1, $Dir2)
% command = ['".' compareDirectoriesPath ';'...     % command definition
%            ' Compare-Directories ' ...            % command call
%             '\"' referencePath '\" ' ...          % reference
%             '\"' convertStringsToChars(resultPath) '\"']; % difference
% 
% % FULL ORIGINAL COMMAND:
% 
% % powershell -ExecutionPolicy Bypass -Command ". 
% % 'C:\Users\mashi\Desktop\RaymondLab\MATLAB\open field photometry script\Compare-Directories.ps1' ; 
% % Compare-Directories 'C:\Users\mashi\Desktop\RaymondLab\MATLAB\open field photometry script\data\315955m1' 'C:\Users\mashi\Desktop\RaymondLab\MATLAB\open field photometry script\data\demo_target_of_file_creation' "
% 
% % run the actual command using Powershell
% [status, cmdout] = system(append( ...
%     'powershell -ExecutionPolicy Bypass -Command ', ...
%     command ...
%     ));
% % if status ~= 0, error occurred.
% if status ~= 0
%     disp "An error occurred - double check path!"; disp(cmdout);
%     isCorrect = false; diff = "";
%     return
% % otherwise, check cmdout. If no difference, it returns 0
% % cast cmdout to string for convenience
% elseif convertCharsToStrings(cmdout) == convertCharsToStrings(['0', newline])
%     isCorrect = true; diff = "";
%     return
% % if cmdout isn't 0, it returns the difference in content
% else
%     isCorrect = false; diff = cmdout;
% end