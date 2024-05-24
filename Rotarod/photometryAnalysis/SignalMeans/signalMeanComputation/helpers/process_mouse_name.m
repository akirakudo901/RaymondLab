%% Author: Akira Kudo
% Processes mouse name in a specific way
% get mouse name, day, trial
function [mouseName,mouseNameTotalTrial,day,trialInDay,totalTrial] = ...
    process_mouse_name(dataDir)
    dataDir = convertStringsToChars(dataDir);
    
    if ~endsWith(dataDir, "\") && ~endsWith(dataDir, "/")
        [~, filename] = fileparts(dataDir(1:end));
    else
        [~, filename] = fileparts(dataDir(1:end-1));
    end

    % create a regular expression to extract mouse name from file name
    % file name is of the form:
    %  - "dayN_MouseName_rotarodM" where N is the day, MouseName is the 
    %    name of the mouse, and M is the trial (possibly including '-')
    % * Occasionally, we have names of the form "[Rr]edo_MouseName_rotarodM".
    %   In this case, return: 
    % - mouseName as MouseName part
    % - mouseNameTotalTrial as Redo_MouseName_M
    % - day as "Redo", trialInDay as M, totalTrial as M
    beforeMouseName = '([Dd]ay[1234]_|[Rr]edo_)'; afterMouseName = '_[rR]otarod\d';
    mouseNameStart = regexp(filename,beforeMouseName,  'end') + 1;
    mouseNameEnd   = regexp(filename, afterMouseName,'start') - 1;
    % guard against files not adhering to this rule
    mouseName = filename(mouseNameStart:mouseNameEnd);
    if isempty(mouseName) 
        mouseName = append("IrregularFileName_", filename); 
    end
    mouseName = erase(mouseName, ["_", "-"]);
    % day
    if startsWith(filename, ["Redo", "redo"])
        day = "Redo";
    else
        day = regexp(filename, '[dD]ay\d*', 'match');
        day = day{1}(4:end);
    end
    % trial
    trialEndIdx = regexp(filename, '_[rR]otarod', 'end'); 
    trialInDay = filename(trialEndIdx + 1:end);

    % deal with situation where trial is of the form "n-m"!
    if contains(trialInDay, '-') || contains(trialInDay, '_')
        % In such a case, n is the number of trial, and m is the iteration
        % within trial, where all but the trial with highest m likely failed.
        % WE ASSUME N AND M ARE BOTH ONE DIGIT
        intTrial = extractBetween(trialInDay, 1, 1);
    else 
        intTrial= trialInDay;
    end
    
    % then calculate the trial number: e.g. day 3 trial 3 = 2*3 + 3 = 9th 
    TRIALS_PER_DAY = 3;
    if startsWith(filename, ["Redo", "redo"])
        totalTrial = intTrial;
        if strlength(totalTrial) == 1
            totalTrial = "0" + totalTrial;
        end
        totalTrial = "Redo_" + totalTrial;
    else
        totalTrial = string( (str2double(day) - 1) * TRIALS_PER_DAY + str2double(intTrial) );
        if strlength(totalTrial) == 1
            totalTrial = "0" + totalTrial;
        end
    end
    
    [mnStr, totalTrial] = convertCharsToStrings(mouseName, totalTrial);
    mouseNameTotalTrial = join([mnStr, totalTrial], "_");
    
    if contains(trialInDay, '-') || contains(trialInDay, '_')
        mouseNameTotalTrial = mouseNameTotalTrial + "_" + trialInDay(end);
    end
end