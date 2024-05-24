
% This script allows you to fill, given a file which holds data for the
% trials analyzed already, the other trials that don't yet have values yet,
% so that each mouse name is followed by exactly 12 trials. 
% Useful to clean up the layout of the csv such that it is loaded cleanly
% to Graphpad Prism.

function fill_empty_trial_entries(FILEPATH)
    addpath(fullfile("..", "utils"));

    disp("====================================")
    disp("Filling empty trials!")
    disp("====================================")
    
    FILENAME_IDX = 1; DAY_IDX = 2; TRIAL_IDX = 3;
    
    [folder, name, ext] = fileparts(FILEPATH);
    csvName = append(name, ext);
    
    TRIAL_PER_DAY = 3;
    MAX_DAY = 4;
    MAX_TRIAL_NUM = TRIAL_PER_DAY * MAX_DAY;
    
    % first read into resultData object
    rdFields = Constants_RotarodPhotometry.Means_ResultDataFields ;
    rd = ResultData(rdFields);
    
    rd.loaddata(FILEPATH);
    
    % also load a csv file
    csvData = readcell(FILEPATH, "Delimiter", ",", ...
        "ConsecutiveDelimitersRule", "split");
    [r,~] = size(csvData);
    
    % for each entry in csvData
    for rowIdx = 2:r
        % If this is is a failed row, skip
        fileName = csvData{rowIdx, FILENAME_IDX};
        if startsWith(fileName, 'Failed_')
            continue
        end
        % if it is part of the redone rows, skip too
        if contains(fileName, ["Redo", "redo"])
            continue
        end
        % check if this entry has a dash in it, and if so, skip
        skippedExpression = '\d_\d{1,2}(_\d)';
        match = regexp(fileName, skippedExpression, 'match');
        if ~isempty(match)
            continue
        end

        % strip file name with day and trial info
        expression = '\d_\d{1,2}(_\d)?';
        endOfRawFileName = regexp(fileName, expression, 'start');
        rawFileName = fileName(1:endOfRawFileName);
        for i = 1:MAX_TRIAL_NUM
            originalNumberedFileName = append(rawFileName, "_", num2str(i));
            if i < 10
                newNumberedFileName = append(rawFileName, "_", "0", num2str(i));
            else 
                newNumberedFileName = originalNumberedFileName;
            end
    
            dayFromi = floor((i - 1)/TRIAL_PER_DAY) + 1;
            trialFromi = i - (dayFromi - 1) * 3;

            % check if the new name is already in ResultData (or possibly
            % an entry with close name, indicating retry for the same
            % trial) and only add entry if there is no such match
            match1 = regexp(rd.getkeys(), originalNumberedFileName, 'match');
            match2 = regexp(rd.getkeys(),      newNumberedFileName, 'match');
            fileWithMatchToOneExists = any(~cellfun(@isempty, match1));
            fileWithMatchToTwoExists = any(~cellfun(@isempty, match2));
            
            if ~fileWithMatchToOneExists && ~fileWithMatchToTwoExists
                rd.adddata(newNumberedFileName, ...
                    struct('Day', dayFromi, ...
                           'Trial', trialFromi', ...
                           'Means_Green', '', ...
                           'Meanshift_Green', '', ...
                           'Means_Red', '', ... 
                           'Meanshift_Red', '', ...
                           'MouseName', rawFileName, ...,
                           'TotalTrial', i, ...
                           'After_TooShort', '', ...
                           'Note_Onset_Size1', '', ...
                           'Note_Onset_Size2', '', ...
                           'PtAB_Onset_Size1', '', ...
                           'PtAB_Onset_Size2', '', ...
                           'Exception', '', ...
                           'Info', 'Filled in as empty trial.'...
                           )...
                           )
            end
        end
    end
    
    % save result
    rd.savedata(folder, csvName);
end