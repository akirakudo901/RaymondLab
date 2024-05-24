ROW_MAX = 446;
FILENAME_IDX = 1; DAY_IDX = 2; TRIAL_IDX = 3;

FILENAME = "photometry_means_analysis_result.csv";

% first read as csv
data = readcell(FILENAME);


rowsToDeleteLater = {};
% for all mouse names:
for idx = 2:ROW_MAX
    % get mouse name, day and trial
    fullMouseName = data{idx, FILENAME_IDX};
    day   = string( data{idx,   DAY_IDX} );
    trial = string( data{idx, TRIAL_IDX} );
    % recreate the correct name (name_day_trial)
    % get mouse name by removing day and trial from it
    mouseNameOnly = fullMouseName(1:end - strlength(day) - strlength(trial));
        
    [mnStr, dayStr, trialStr] = convertCharsToStrings(mouseNameOnly, day, trial);
    expectedName = join([mnStr, dayStr, trialStr], "_");

    % check if mouse name comforms to naming scheme already, and if so skip
    if fullMouseName == expectedName
        continue
    % otherwise check if name is of old style to be removed
    elseif ~endsWith(fullMouseName, append(trial, day))
        % if not, remember to remove the row later
        rowsToDeleteLater{end + 1} = idx;
    % otherwise we rename the entry correctly
    else
        % readjust name
        data{idx,1} = expectedName;
    end
end

% remove all entries that are set to b removed
for i = length(rowsToDeleteLater):-1:1
    removedRowNum = rowsToDeleteLater{i};
    data(removedRowNum, :) = [];
end

% save result
writecell(data, FILENAME);