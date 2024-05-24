FILENAME_IDX = 1; DAY_IDX = 2; TRIAL_IDX = 3;

FILENAME = fullfile(".", "results", "photometry_means_analysis_result.csv");

% first read as csv
data = readcell(FILENAME);
[r, c] = size(data);

% for all mouse names:
for idx = 2:r
    % get mouse name, day and trial
    fullMouseName = data{idx, FILENAME_IDX};
    day   = string( data{idx,   DAY_IDX} );
    trial = string( data{idx, TRIAL_IDX} ); 
    % recreate the correct name (name_day_trial)
    % get mouse name by removing day and trial from it
    mouseNameOnly = fullMouseName(1:end - strlength(day) - strlength(trial) - 2);
    
    % deal with situation where trial is of the form "n-m"!
    if contains(trial, '-') || contains(trial, '_')
        % In such a case, n is the number of trial, and m is the iteration
        % within trial, where all but the trial with highest m likely failed.
        % WE ASSUME N AND M ARE BOTH ONE DIGIT
        trial = extractAfter(extractBefore(trial, 2), 0);
    end
    
    % then calculate the trial number: e.g. day 3 trial 3 = 2*3 + 3 = 9th 
    TRIALS_PER_DAY = 3;
    overallTrialNumber = string( (str2double(day) - 1) * TRIALS_PER_DAY + str2double(trial) );

    [mnStr, overallTrialNumber] = convertCharsToStrings(mouseNameOnly, overallTrialNumber);
    expectedName = join([mnStr, overallTrialNumber], "_");

    
    % readjust name
    data{idx,1} = expectedName;
end

% Remove duplicates based on "n-m"!
% Hence, 1) Go through all cases where the name is "n-m"
% 2) Check what highest m value is and note it down
% 3) Remove any other entries with the same n value
% 4) Rename n-m to n.


% save result
writecell(data, FILENAME);